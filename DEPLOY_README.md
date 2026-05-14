# ComfyUI on Cloud Run with Google GenMedia custom nodes

Deploys [ComfyUI](https://github.com/comfyanonymous/ComfyUI) to Google Cloud
Run with the
[comfyui-google-genmedia-custom-nodes](https://github.com/SunilKumarJB/comfyui-google-genmedia-custom-nodes)
preinstalled, so you get nodes for Gemini, Imagen 3/4, Veo 2/3.1, Lyria,
Chirp TTS, and Virtual Try-On out of the box.

## Why this works on Cloud Run (no GPU needed)

The custom nodes don't run diffusion models locally — every node calls a
**Vertex AI API**. ComfyUI is just orchestrating HTTP calls. So a CPU-only
Cloud Run service is enough; the heavy compute happens on Google's side and
you pay for it via Vertex AI pricing, not Cloud Run GPU time.

If you also want to run *local* models (e.g. SDXL checkpoints) alongside
the Google nodes, you'll need GPU — see "Adding GPU support" below.

## Files

- `Dockerfile` — builds ComfyUI + the custom nodes into one image
- `deploy.sh` — end-to-end GCP setup
- `.dockerignore` — keeps the build context small

## Quick start

```bash
gcloud auth login
gcloud config set project my-gcp-project    # whatever this is set to is what gets deployed
chmod +x deploy.sh
./deploy.sh
or 
ALLOW_UNAUTHENTICATED=true ./deploy.sh # Public Cloud Run URL
```

The script picks up the **deploy-time** project from `gcloud config
get-value project` and the Cloud Run region from `gcloud config
get-value run/region` (falling back to `compute/region`, then
`us-central1`). It prints the resolved values and asks for
confirmation before creating any resources.

The **runtime** env vars (`GCP_PROJECT_ID`, `GCP_REGION`,
`EMBEDDING_REGION`) are *not* auto-derived from those — they're only
set in the container if you pass them explicitly:

```bash
# Most common: pin Vertex AI calls to a specific project+region
GCP_PROJECT_ID=my-proj GCP_REGION=us-central1 ./deploy.sh

# Override Cloud Run region too (where ComfyUI runs)
REGION=europe-west1 GCP_PROJECT_ID=my-proj GCP_REGION=global EMBEDDING_REGION=us-central1 ./deploy.sh

# Bare minimum — fill in gcp_project_id/gcp_region per-node in the UI
./deploy.sh
```

Before any APIs are enabled or resources created, the script prints the
resolved project + region and asks you to confirm — so a forgotten
`gcloud config set project` doesn't accidentally deploy to the wrong place.

The first build takes ~8–15 minutes. Subsequent deploys are faster.

## Accessing ComfyUI

By default the script deploys with `--no-allow-unauthenticated` because
ComfyUI has no built-in auth — leaving it public means anyone who finds
the URL can run workflows on your Vertex AI quota and see every other
user's outputs.

To open it locally:

```bash
gcloud run services proxy comfyui --region=us-central1
# then visit http://localhost:8080
```

To give a teammate access without making it public:

```bash
gcloud run services add-iam-policy-binding comfyui \
    --region=us-central1 \
    --member="user:teammate@example.com" \
    --role="roles/run.invoker"
```

If you really want it public (demo, throwaway project):

```bash
ALLOW_UNAUTHENTICATED=true ./deploy.sh
```

## Configuration knobs

All overridable via env vars before running `deploy.sh`:

| Var | Default | Notes |
|---|---|---|
| `PROJECT_ID` | active gcloud project | Auto-detected from `gcloud config get-value project`. Override only for one-off cross-project deploys. *(this is for the deploy itself — building, granting IAM, etc.)* |
| `REGION` | gcloud `run/region` → `compute/region` → `us-central1` | Cloud Run region — where ComfyUI **runs**. |
| `GCP_PROJECT_ID` | *(unset)* | Runtime env var passed to the container. Where Vertex AI calls are **billed**. Only set if you pass it. |
| `GCP_REGION` | `global` | Runtime env var passed to the container. Vertex AI region for Imagen/Veo/etc. Only set if you pass it. |
| `EMBEDDING_REGION` | `us-central1` | Runtime env var for Vertex AI embedding endpoints. Only set if you pass it. |
| `SERVICE_NAME` | `comfyui` | Cloud Run service name |
| `MEMORY` | `4Gi` | Bump if ComfyUI OOMs on large workflows |
| `CPU` | `2` | |
| `TIMEOUT` | `3600` | Cloud Run max is 3600s (1 hour) |
| `MIN_INSTANCES` | `1` | See warning below; 0 will kill in-flight jobs |
| `MAX_INSTANCES` | `3` | |
| `ALLOW_UNAUTHENTICATED` | `false` | See warning above |

### Why `MIN_INSTANCES=1` is the default (and `--no-cpu-throttling` is on)

ComfyUI processes workflows on a background worker thread that runs
*between* HTTP requests — it polls Vertex AI for video job completion,
pushes progress over websockets, and writes outputs to disk after the
HTTP request that submitted the job has long since returned.

Cloud Run's default request-based billing throttles CPU to near-zero
between requests, which **pauses that worker thread mid-job**. The fix is
either:

- `MIN_INSTANCES=1` + `--no-cpu-throttling` (this script's default) —
  one always-on instance, costs more, but jobs run reliably.
- `MIN_INSTANCES=0` + accept that any job in flight when the websocket
  disconnects gets orphaned. Cheaper, fragile.

Setting `MIN_INSTANCES=0` here without removing `--no-cpu-throttling`
also doesn't make sense: there's no instance for "always-on CPU" to apply
to until a request arrives, and you'll still get cold-start kills.

## Using the Google AI nodes

### How env vars get to the container

The `comfyui-google-genmedia-custom-nodes` repo ships an `.env.example`
pointing at four variables: `GCP_PROJECT_ID`, `GCP_REGION`,
`EMBEDDING_REGION`, and `GEMINI_API_KEY`. Locally those would go in a
`.env` file. On Cloud Run we don't need a file — the nodes use
`python-dotenv`, which reads variables from the process environment
first and only falls back to `.env` if a variable isn't already set.

So `--set-env-vars` on `gcloud run deploy` is functionally identical to
having the value in a `.env` file, and that's what the deploy script
uses.

### Explicit opt-in (no defaults, no inference)

The deploy script **does not assume any value for the runtime env
vars**. It passes through *only* what you explicitly set when invoking
it. If you don't pass a variable, it doesn't appear in the container —
the custom nodes will rely on their per-node UI fields instead.

```bash
# All three: project, Vertex AI region, embedding region
GCP_PROJECT_ID=my-proj GCP_REGION=us-central1 EMBEDDING_REGION=us-central1 ./deploy.sh

# Just the project (most common — let nodes pick region per-call)
GCP_PROJECT_ID=my-proj ./deploy.sh

# Nothing — fully relies on the per-node UI fields
./deploy.sh
```

This is intentional. Cloud Run's region (where ComfyUI runs) and
Vertex AI's region (where models are served) are independent choices.
Picking one based on the other would silently fail when the model
isn't available in the assumed region. Better to require an explicit
value or let the per-node UI fields handle it.

On *redeploys*, if you don't pass any of the runtime env vars, the
script uses `--clear-env-vars` so a previous revision's values don't
linger on the new revision.

### Per-node fields are still authoritative

Every Google AI node has its own `gcp_project_id` and `gcp_region` UI
fields. When those are filled in, they win — the env vars only serve
as fallbacks. Either way, fill them in on every Google AI node in your
workflow; that's the path the upstream README documents.

### About `GEMINI_API_KEY`

The script *deliberately won't inject* `GEMINI_API_KEY` even if you
export it. `--set-env-vars` stores the value as plaintext on the
revision spec, where anyone with `roles/run.developer` can read it.
If you need to use the AI Studio API path (instead of Vertex AI), put
the key in Secret Manager and add this to the `gcloud run deploy`
call manually:

```
--update-secrets=GEMINI_API_KEY=projects/PROJECT/secrets/gemini-key:latest
```

The default Vertex AI path doesn't need a key at all — the runtime
service account handles auth.

### Service account and permissions

The runtime service account (`comfyui-runtime@…`) gets:
- `roles/aiplatform.user` — call Vertex AI
- `roles/storage.objectUser` — needed for Veo's lossless GCS output

Authentication uses Application Default Credentials, which on Cloud Run
automatically resolve to the attached service account via the metadata
server. No keys, no `gcloud auth`, no `GOOGLE_APPLICATION_CREDENTIALS`
to set.

If you'll use lossless Veo workflows, create a GCS bucket in the same
project and point `output_gcs_uri` at it (`gs://your-bucket/path`).

## Important caveats

**No persistent storage.** Cloud Run instance disks are ephemeral. Models,
generated images saved to the local `output/` dir, and the asset-manager
SQLite DB all disappear when the instance restarts. If you need
persistence:
- Send Veo videos straight to GCS via `output_gcs_uri`
- Mount a GCS volume (Cloud Run gen2 supports this) at
  `/app/ComfyUI/output` — see the Cloud Run docs on Cloud Storage volumes
- Or move to GKE with a Filestore mount

**60-minute hard ceiling on requests.** Cloud Run's max request timeout
is 3600s. The websocket that streams progress to the UI is bound by this.
Veo lossless or large-batch jobs can take longer — they'll still complete
on Vertex AI's side and write to your GCS bucket, but the UI websocket
will drop. Use `output_gcs_uri` and check the bucket directly for those.

**Cold starts (~30–60s).** Even with `MIN_INSTANCES=1`, scaling up under
load produces cold starts. ComfyUI imports a lot of Python at boot. The
deploy uses `--cpu-boost` to halve cold-start time.

**One workflow at a time per instance.** ComfyUI's queue is single-
threaded. `CONCURRENCY=4` lets the websocket + HTTP UI requests share the
instance with the worker, but parallel runs need `MAX_INSTANCES` headroom
*and* multiple users hitting the UI — Cloud Run won't auto-shard a single
user's queue across instances.

## Adding GPU support

Cloud Run supports NVIDIA L4 GPUs (GA as of 2024). To use them:

1. Change the Dockerfile base to a CUDA image
   (e.g. `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`), install Python,
   and drop the `--cpu` flag in the `CMD`.
2. Install the matching CUDA torch wheels instead of the CPU index.
3. Add to the `gcloud run deploy` call:
   ```
   --gpu=1 --gpu-type=nvidia-l4
   ```
4. Bump `MEMORY` to at least `16Gi` and `CPU` to `8`.
5. GPU isn't available in every region — check the Cloud Run GPU docs
   before changing `REGION`.

## Changelog from the first cut of this script

After review, several issues from the initial version were fixed:

1. **Pinned ComfyUI version was years out of date** (`v0.3.27` → `v0.3.65`).
   The old tag still resolves but misses many fixes; the actual current
   release is `v0.20.x` but `v0.3.x` line is what most stable custom-node
   setups still target — bump as needed.
2. **Removed the bogus `--disable-auto-launch` flag** from the ComfyUI
   command line. ComfyUI's CLI doesn't have that flag; it would have
   crashed on startup. (ComfyUI also doesn't auto-open a browser when
   running headless.)
3. **Added `--no-cpu-throttling`** to fix the silent worker-thread freeze
   under request-based CPU billing — the most impactful bug.
4. **Changed `MIN_INSTANCES` default from 0 → 1** to match the always-on
   CPU model and avoid cold-start kills mid-job.
5. **Added Cloud Build default-SA permission grants** — projects created
   after July 2024 don't get these automatically and would fail at the
   `gcloud builds submit` step with a permissions error.
6. **Added the `iam.serviceAccountUser` grant** to the deploying user, so
   non-Owner accounts can attach the runtime SA to the service.
7. **Made sample-workflow copy fail loudly** instead of silently swallowing
   errors with `2>/dev/null || true`.
8. **Bumped Python to 3.12** (3.11 worked, but ComfyUI explicitly
   recommends 3.12+).
9. **Removed unneeded apt packages** (`libsm6`, `libxext6`, `libxrender1`)
   that were only relevant for the full `opencv-python`; the custom
   nodes use `opencv-python-headless`.
10. **Added gcloud auth + Dockerfile presence checks** at the top of the
    script so failures are clearer.

## Troubleshooting

- **Build fails with permission denied pushing to Artifact Registry:**
  fixed by item 5 above. If it still fails, check that the
  `${PROJECT_NUMBER}-compute@developer.gserviceaccount.com` SA has
  `roles/artifactregistry.writer` on your project.
- **Nodes don't appear in ComfyUI:** check Cloud Run logs for an import
  error from `comfyui-google-genmedia-custom-nodes`. Most often it's a
  Vertex AI auth problem — confirm the runtime SA has
  `roles/aiplatform.user`.
- **403 from Vertex AI:** the model you're calling may not be available
  in your `REGION`. Try `us-central1` or `global` (some Gemini nodes
  use the global endpoint).
- **Workflow appears to hang at 50%:** classic CPU-throttling symptom.
  Confirm the deploy used `--no-cpu-throttling` (visible in the
  service revision under "Always allocate CPU = true").
