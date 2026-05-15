#!/usr/bin/env bash
# =============================================================================
# Deploy ComfyUI + Google GenMedia custom nodes to Cloud Run
# =============================================================================
# What this does:
#   1. Verifies gcloud auth and project
#   2. Enables required GCP APIs
#   3. Creates an Artifact Registry repo for the container image
#   4. Creates a runtime service account with Vertex AI + GCS access
#   5. Grants the deploying user permission to act as that runtime SA
#   6. Grants the Cloud Build default SA the perms it needs (required for
#      projects created after July 2024, where this is no longer automatic)
#   7. Builds the image with Cloud Build
#   8. Deploys it to Cloud Run with always-on CPU + a 1hr timeout
#
# Prereqs:
#   - gcloud CLI installed and authenticated (`gcloud auth login`)
#   - A billing-enabled GCP project
#   - Dockerfile in the same directory as this script
# =============================================================================

set -euo pipefail

# -------- Configuration (override via env vars) ------------------------------
# PROJECT_ID falls back to whatever's set in `gcloud config get-value project`.
# That way `gcloud config set project foo && ./deploy.sh` just works, and you
# can still override with `PROJECT_ID=bar ./deploy.sh` for a one-off deploy
# to a different project without changing your gcloud config.
if [[ -z "${PROJECT_ID:-}" ]]; then
    PROJECT_ID="$(gcloud config get-value project 2>/dev/null || true)"
fi
if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
    echo "ERROR: No project ID found." >&2
    echo "Either run: gcloud config set project YOUR_PROJECT_ID" >&2
    echo "Or run:     PROJECT_ID=your-project ./deploy.sh" >&2
    exit 1
fi

# REGION falls back to gcloud's configured Cloud Run region, then compute
# region, then us-central1. Same override semantics as PROJECT_ID:
# `REGION=europe-west1 ./deploy.sh` wins over everything.
if [[ -z "${REGION:-}" ]]; then
    REGION="$(gcloud config get-value run/region 2>/dev/null || true)"
fi
if [[ -z "${REGION}" || "${REGION}" == "(unset)" ]]; then
    REGION="$(gcloud config get-value compute/region 2>/dev/null || true)"
fi
if [[ -z "${REGION}" || "${REGION}" == "(unset)" ]]; then
    REGION="us-central1"
fi

SERVICE_NAME="${SERVICE_NAME:-comfyui}"
REPO_NAME="${REPO_NAME:-comfyui-repo}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
SA_NAME="${SA_NAME:-comfyui-runtime}"

# ----- Custom-node env vars (explicit opt-in, no defaults) -----
# These are passed straight through to the running container as the
# Vertex AI configuration the GenMedia custom nodes read at startup.
#
# Design choice: NONE of these have defaults. We don't reuse the Cloud
# Run REGION for GCP_REGION, and we don't pick a region for embeddings.
# If you don't pass them, they don't get set in the container — the
# nodes will rely on their per-node UI fields (gcp_project_id /
# gcp_region) instead, which is what the upstream README assumes anyway.
#
# Why no defaults? The Cloud Run region (where ComfyUI runs) and the
# Vertex AI region (where models are served) are independent choices.
# Picking one based on the other is a guess, and a wrong guess fails
# silently — Vertex returns 404s for models not available in that
# region. Better to require an explicit value or no value at all.
#
# Pass them via env vars when invoking the script:
#   GCP_PROJECT_ID=my-proj GCP_REGION=us-central1 ./deploy.sh
#
# All four are optional. Leave any of them unset to skip injection.
GCP_PROJECT_ID="${GCP_PROJECT_ID:-}"
GCP_REGION="${GCP_REGION:-global}"
EMBEDDING_REGION="${EMBEDDING_REGION:-us-central1}"
GEMINI_API_KEY="${GEMINI_API_KEY:-}"

# Resource sizing. ComfyUI is just orchestrating Vertex AI calls here, but
# it loads its full Python stack into memory and the model-management code
# allocates generously even on CPU.
MEMORY="${MEMORY:-16Gi}"
CPU="${CPU:-8}"

# Cloud Run's max request timeout is 3600 seconds (1 hour). Veo lossless
# or large batch jobs may exceed this — see the README for caveats.
TIMEOUT="${TIMEOUT:-3600}"

# Concurrency: ComfyUI runs one workflow at a time per instance via its
# internal queue; this number governs the websocket + UI HTTP requests
# that share the instance with that worker.
CONCURRENCY="${CONCURRENCY:-4}"

# IMPORTANT: keep MIN_INSTANCES at 1. ComfyUI's queue runs in a background
# thread that ticks between HTTP requests (polling Vertex AI for video job
# completion, pushing progress over websockets, etc.). With MIN_INSTANCES=0
# the instance gets reaped shortly after the last request, killing any
# in-flight generation. Always-on CPU (--no-cpu-throttling, set below)
# also requires at least 1 always-running instance to be useful.
MIN_INSTANCES="${MIN_INSTANCES:-1}"
MAX_INSTANCES="${MAX_INSTANCES:-3}"

# Whether to allow unauthenticated access. STRONGLY recommend "false" —
# ComfyUI has no built-in auth and exposing it publicly lets anyone burn
# your Vertex AI quota and read everyone else's outputs.
ALLOW_UNAUTHENTICATED="${ALLOW_UNAUTHENTICATED:-false}"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:${IMAGE_TAG}"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# -------- Helpers ------------------------------------------------------------
log() { echo -e "\n\033[1;34m==>\033[0m $*"; }
err() { echo -e "\n\033[1;31m!!\033[0m $*" >&2; exit 1; }

# -------- 0. Sanity checks ---------------------------------------------------
command -v gcloud >/dev/null || err "gcloud CLI not found. Install from https://cloud.google.com/sdk/docs/install"

ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null || true)
[[ -z "${ACTIVE_ACCOUNT}" ]] && err "No active gcloud account. Run: gcloud auth login"

[[ -f Dockerfile ]] || err "Dockerfile not found in current directory. cd to the deployment directory first."

log "Deploying as: ${ACTIVE_ACCOUNT}"
log "Target project: ${PROJECT_ID}"
log "Target region:  ${REGION}"

# Last chance to bail before we start enabling APIs and creating resources.
# Skip the prompt in non-interactive shells (CI) — the explicit env vars are
# the contract there.
if [[ -t 0 ]]; then
    read -r -p "Proceed with deploy? [y/N] " confirm
    [[ "${confirm,,}" == "y" || "${confirm,,}" == "yes" ]] || err "Aborted by user."
fi

# -------- 1. Project + APIs --------------------------------------------------
log "Setting active project to ${PROJECT_ID}"
gcloud config set project "${PROJECT_ID}" --quiet

PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')

log "Enabling required APIs (this can take a minute)"
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    aiplatform.googleapis.com \
    storage.googleapis.com \
    iam.googleapis.com \
    --quiet

# -------- 2. Artifact Registry ----------------------------------------------
log "Creating Artifact Registry repo '${REPO_NAME}' in ${REGION} (idempotent)"
if ! gcloud artifacts repositories describe "${REPO_NAME}" \
        --location="${REGION}" >/dev/null 2>&1; then
    gcloud artifacts repositories create "${REPO_NAME}" \
        --repository-format=docker \
        --location="${REGION}" \
        --description="ComfyUI container images"
else
    echo "    Repo already exists, continuing."
fi

# -------- 3. Runtime service account ----------------------------------------
log "Creating runtime service account '${SA_NAME}' (idempotent)"
if ! gcloud iam service-accounts describe "${SA_EMAIL}" >/dev/null 2>&1; then
    gcloud iam service-accounts create "${SA_NAME}" \
        --display-name="ComfyUI Cloud Run runtime"
else
    echo "    Service account already exists, continuing."
fi

log "Granting Vertex AI Admin + Storage Object Admin + Bigquery Admin to ${SA_EMAIL}"
# aiplatform.user → call Vertex AI APIs (Imagen, Veo, Gemini, etc.)
# storage.objectUser → read/write GCS buckets used by Veo for lossless output
for ROLE in roles/aiplatform.admin roles/storage.admin roles/bigquery.admin; do
    gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="${ROLE}" \
        --condition=None \
        --quiet >/dev/null
done

# Without this, a non-Owner deployer can't attach the runtime SA to the
# Cloud Run service (`gcloud run deploy --service-account=...` fails).
log "Granting deployer iam.serviceAccountUser on ${SA_EMAIL}"
gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
    --member="user:${ACTIVE_ACCOUNT}" \
    --role="roles/iam.serviceAccountUser" \
    --quiet >/dev/null || \
    echo "    (Skipped — likely a service account or org-managed identity. Grant manually if deploy fails.)"

# -------- 4. Cloud Build SA permissions -------------------------------------
# Since July 2024, new GCP projects no longer auto-grant the default
# Compute Engine SA the broad Cloud Build permissions it used to have. As
# a result the legacy `gcloud builds submit` flow can fail with confusing
# "permission denied" errors on fresh projects. We grant the minimum
# explicitly, idempotently.
CB_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
log "Ensuring Cloud Build SA (${CB_SA}) has build + push perms"
for ROLE in \
    roles/logging.logWriter \
    roles/artifactregistry.writer \
    roles/storage.objectViewer; do
    gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
        --member="serviceAccount:${CB_SA}" \
        --role="${ROLE}" \
        --condition=None \
        --quiet >/dev/null
done

# -------- 5. Build the image with Cloud Build -------------------------------
log "Submitting build to Cloud Build → ${IMAGE_URI}"
log "(This usually takes 8–15 minutes the first time; subsequent builds are faster)"
gcloud builds submit \
    --tag "${IMAGE_URI}" \
    --region="${REGION}" \
    .

# -------- 6. Deploy to Cloud Run --------------------------------------------
AUTH_FLAG="--no-allow-unauthenticated"
if [[ "${ALLOW_UNAUTHENTICATED}" == "true" ]]; then
    AUTH_FLAG="--allow-unauthenticated"
    cat <<'EOF'

╔══════════════════════════════════════════════════════════════════════════╗
║  WARNING: Deploying with PUBLIC access. ComfyUI has no built-in auth.   ║
║  Anyone with the URL can run workflows on your Vertex AI quota and see  ║
║  every other user's outputs. Press Ctrl-C now if this is unintended.    ║
╚══════════════════════════════════════════════════════════════════════════╝

EOF
    sleep 5
fi

log "Deploying to Cloud Run service '${SERVICE_NAME}'"

# Build the --set-env-vars list from whatever the user explicitly passed.
# If the user didn't pass a variable, we don't set it — the custom nodes
# fall back to their per-node UI fields.
ENV_VARS=()
[[ -n "${GCP_PROJECT_ID}"   ]] && ENV_VARS+=("GCP_PROJECT_ID=${GCP_PROJECT_ID}")
[[ -n "${GCP_REGION}"       ]] && ENV_VARS+=("GCP_REGION=${GCP_REGION}")
[[ -n "${EMBEDDING_REGION}" ]] && ENV_VARS+=("EMBEDDING_REGION=${EMBEDDING_REGION}")
# Note: we deliberately do NOT inject GEMINI_API_KEY here — putting it in
# --set-env-vars saves it as plaintext on the revision spec, visible to
# anyone with run.developer. If you need it, plumb it via Secret Manager
# (`--update-secrets=GEMINI_API_KEY=projects/.../secrets/gemini-key:latest`).

DEPLOY_ARGS=(
    "${SERVICE_NAME}"
    --image="${IMAGE_URI}"
    --region="${REGION}"
    --platform=managed
    --service-account="${SA_EMAIL}"
    --memory="${MEMORY}"
    --cpu="${CPU}"
    --timeout="${TIMEOUT}"
    --concurrency="${CONCURRENCY}"
    --min-instances="${MIN_INSTANCES}"
    --max-instances="${MAX_INSTANCES}"
    --port=8080
    --execution-environment=gen2
    --no-cpu-throttling
    --cpu-boost
    "${AUTH_FLAG}"
)

if (( ${#ENV_VARS[@]} > 0 )); then
    # Join with commas. printf+IFS avoids the "trailing comma" footgun.
    JOINED=$(IFS=, ; echo "${ENV_VARS[*]}")
    DEPLOY_ARGS+=(--set-env-vars="${JOINED}")
    log "Injecting env vars: ${JOINED}"
else
    # On a redeploy, --clear-env-vars wipes any previously-set values so
    # the running service matches what the user actually asked for.
    DEPLOY_ARGS+=(--clear-env-vars)
    log "No GCP_*/EMBEDDING_REGION passed; container env will be empty."
    log "Custom nodes will use their per-node gcp_project_id/gcp_region UI fields."
fi

# Notes on the flags above:
#   --no-cpu-throttling   = always-on CPU. ComfyUI's queue runs in a
#                           background thread that ticks between HTTP
#                           requests; with the default request-based
#                           billing, that thread gets throttled mid-job.
#   --cpu-boost           = doubles CPU during cold start so ComfyUI's
#                           ~30-60s import-time doesn't time out.
#   --execution-environment=gen2 = required for some features (mounting
#                           GCS volumes, etc.) and slightly faster boot.
gcloud run deploy "${DEPLOY_ARGS[@]}"

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region="${REGION}" \
    --format='value(status.url)')

log "Deployment complete!"
echo
echo "Service URL: ${SERVICE_URL}"
echo

if [[ "${ALLOW_UNAUTHENTICATED}" != "true" ]]; then
    cat <<EOF
The service is IAM-protected. To open ComfyUI in your browser:

    gcloud run services proxy ${SERVICE_NAME} --region=${REGION}

Then go to http://localhost:8080 — gcloud authenticates your local
user against Cloud Run automatically.

To grant another user direct browser access (still authenticated):

    gcloud run services add-iam-policy-binding ${SERVICE_NAME} \\
        --region=${REGION} \\
        --member="user:someone@example.com" \\
        --role="roles/run.invoker"

EOF
fi

echo
if (( ${#ENV_VARS[@]} > 0 )); then
    echo "Container env vars set on this revision:"
    for kv in "${ENV_VARS[@]}"; do
        echo "  $kv"
    done
else
    echo "No container env vars were set on this revision."
fi

cat <<EOF

Inside ComfyUI, every Google AI node has its own gcp_project_id and
gcp_region UI fields — fill them in on each node in your workflow.
Those node fields are authoritative; the env vars above (if any) only
serve as defaults the node code falls back to when its UI field is
blank.

Cost note: with MIN_INSTANCES=${MIN_INSTANCES} and --no-cpu-throttling,
you're paying for ${MIN_INSTANCES} always-on instance(s). If you want to
scale to zero (cheaper, but cold starts kill in-flight jobs), redeploy
with MIN_INSTANCES=0 — but only if you accept that limitation.
EOF
