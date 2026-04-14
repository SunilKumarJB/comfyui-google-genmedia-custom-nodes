# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import time
from typing import Optional

import torch
from google import genai
from google.genai import types

from .asset_db import get_all_assets, insert_or_update_asset, search_assets_by_tags, search_similar_assets
from .base import VertexAIClient
from .constants import GEMINI_USER_AGENT, GeminiEmbeddingModel, GeminiModel
from .custom_exceptions import APIExecutionError, ConfigurationError
from .logger import get_node_logger
from .retry import api_error_retry
from .utils import tensor_to_pil_to_bytes

logger = get_node_logger(__name__)

MEDIA_DIR = os.path.join("output", "asset_manager", "media")
os.makedirs(MEDIA_DIR, exist_ok=True)


class GeminiAssetIndexer(VertexAIClient):
    """
    Indexes an image/video into the local AI asset discovery SQLite database.
    """

    def __init__(
        self,
        gcp_project_id: Optional[str] = None,
        gcp_region: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(
            gcp_project_id=gcp_project_id,
            gcp_region=gcp_region,
            user_agent=GEMINI_USER_AGENT,
            api_key=api_key,
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "custom_tags": ("STRING", {"default": "genai, colorful", "multiline": False}),
            },
            "optional": {
                "storage_mode": (["local", "gcs_bq"], {"default": "local"}),
                "gcs_bucket": ("STRING", {"default": ""}),
                "bq_dataset": ("STRING", {"default": "comfyui_assets"}),
                "bq_table": ("STRING", {"default": "media_index"}),
                "gcp_project_id": ("STRING", {"default": ""}),
                "gcp_region": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "asset_path")
    FUNCTION = "index_asset"
    CATEGORY = "Google AI/Asset Management"

    @api_error_retry
    def index_asset(
        self,
        image: torch.Tensor,
        custom_tags: str,
        storage_mode: str = "local",
        gcs_bucket: str = "",
        bq_dataset: str = "comfyui_assets",
        bq_table: str = "media_index",
        gcp_project_id: str = "",
        gcp_region: str = "",
        api_key: str = "",
    ):
        # If overrides are provided, re-init client
        if gcp_project_id or gcp_region or api_key:
            self.__init__(
                gcp_project_id=gcp_project_id or None,
                gcp_region=gcp_region or None,
                api_key=api_key or None,
            )

        logger.info("Indexing asset using Gemini models...")

        # Convert image to bytes and save locally so UI can render it
        png_bytes = tensor_to_pil_to_bytes(image, format="PNG")
        timestamp = int(time.time())
        rand_id = random.randint(1000, 9999)
        filename = f"asset_{timestamp}_{rand_id}.png"
        filepath = os.path.join(MEDIA_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(png_bytes)

        # 1. Generate Tags and Caption with Gemini 3.1 Pro
        pro_model = GeminiModel.GEMINI_3_1_PRO.value
        prompt = (
            "Analyze this image visually. Return ONLY a valid JSON object with two properties: "
            "'caption' (a detailed visual description) and 'tags' (a comma-separated string of 5 descriptive keywords)."
        )

        image_part = types.Part.from_bytes(data=png_bytes, mime_type="image/png")
        contents = [prompt, image_part]

        try:
            logger.info(f"Calling {pro_model} for structured visual description...")
            pro_res = self.client.models.generate_content(
                model=pro_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                )
            )
            import json
            res_data = json.loads(pro_res.text.strip())
            caption = res_data.get("caption", "Generated image.")
            tags = f"{custom_tags}, {res_data.get('tags', '')}"
        except Exception as e:
            logger.error(f"Error during captioning: {e}")
            caption = "Caption generation failed."
            tags = custom_tags

        # 2. Generate Embedding with Gemini Embedding 2
        emb_model = GeminiEmbeddingModel.GEMINI_EMBEDDING_2.value
        embedding_list = None
        effective_api_key = os.environ.get("GEMINI_API_KEY")
        project_id = os.environ.get("GCP_PROJECT_ID")
        embedded_region = os.environ.get("EMBEDDING_REGION")
        
        try:
            logger.info(f"Calling {emb_model} for multimodal embedding...")
            if effective_api_key:
                cl = genai.Client(api_key=effective_api_key)
            else:
                cl = genai.Client(vertexai=True, project=project_id, location=embedded_region)
            
            emb_res = cl.models.embed_content(
                model=emb_model,
                contents=image_part,
            )
            if emb_res.embeddings and len(emb_res.embeddings) > 0:
                embedding_list = emb_res.embeddings[0].values
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")

        # 3. Save to Database
        try:
            if storage_mode == "gcs_bq":
                from .asset_db import insert_asset_cloud
                if not gcs_bucket:
                    raise ValueError("GCS Bucket required for cloud mode.")
                insert_asset_cloud(
                    file_bytes=png_bytes,
                    filename=filename,
                    filetype="image/png",
                    tags=[t.strip() for t in tags.split(",") if t.strip()],
                    caption=caption,
                    embedding_list=embedding_list,
                    gcs_bucket=gcs_bucket,
                    bq_dataset=bq_dataset,
                    bq_table=bq_table,
                    project_id=self.project_id if hasattr(self, "project_id") else gcp_project_id,
                )
            else:
                insert_or_update_asset(
                    filepath=filepath,
                    filetype="image/png",
                    tags=tags,
                    caption=caption,
                    embedding_list=embedding_list,
                )
            logger.info(f"Successfully saved asset {filename} in {storage_mode} mode.")
        except Exception as e:
            logger.error(f"Error saving to storage ({storage_mode}): {e}")

        return (image, filepath)


# ==============================================================================
# Custom Server Endpoint Hooks
# ==============================================================================
try:
    from aiohttp import web
    from server import PromptServer

    if hasattr(PromptServer, "instance") and PromptServer.instance:
        routes = PromptServer.instance.routes

        @routes.get("/google_genmedia/asset_manager/view")
        async def custom_view_asset(request):
            path = request.query.get("filepath", "")
            if not path:
                return web.Response(status=400, text="Missing filepath")
            
            content_type = "image/png"
            if path.endswith(".mp4") or "video/" in path:
                content_type = "video/mp4"
            
            try:
                if path.startswith("gs://"):
                    from google.cloud import storage
                    bucket_name, blob_name = path[5:].split("/", 1)
                    client = storage.Client()
                    bucket = client.bucket(bucket_name)
                    blob = bucket.blob(blob_name)
                    content = blob.download_as_bytes()
                    return web.Response(body=content, content_type=content_type)
                else:
                    import folder_paths
                    if not os.path.exists(path):
                        root = os.path.dirname(folder_paths.get_output_directory())
                        abs_path = os.path.join(root, path)
                        if os.path.exists(abs_path):
                            path = abs_path
                        else:
                            return web.Response(status=404, text=f"File not found: {path}")
                    with open(path, "rb") as f:
                        content = f.read()
                    return web.Response(body=content, content_type=content_type)
            except Exception as e:
                return web.Response(status=500, text=str(e))

        @routes.get("/google_genmedia/asset_manager/list")
        async def list_assets(request):
            page = int(request.query.get("page", "1"))
            per_page = int(request.query.get("per_page", "10"))
            storage_mode = request.query.get("storage_mode", "local")
            offset = (page - 1) * per_page
            assets = get_all_assets(limit=per_page, offset=offset, storage_mode=storage_mode, project_id=os.environ.get("GCP_PROJECT_ID"))
            from urllib.parse import quote

            for item in assets:
                path = item.get("filepath", "")
                if not path:
                    continue
                item["view_url"] = f"/google_genmedia/asset_manager/view?filepath={quote(path, safe='')}"

            return web.json_response({"assets": assets})

        @routes.post("/google_genmedia/asset_manager/search")
        async def search_assets(request):
            data = await request.json()
            query = data.get("query", "")
            search_mode = data.get("mode", "semantic")
            storage_mode = data.get("storage_mode", "local")

            if search_mode == "tags":
                tags = [t.strip() for t in query.split(",") if t.strip()]
                results = search_assets_by_tags(tags, storage_mode=storage_mode, project_id=os.environ.get("GCP_PROJECT_ID"))
            else:
                effective_api_key = os.environ.get("GEMINI_API_KEY")
                project_id = os.environ.get("GCP_PROJECT_ID")
                region = os.environ.get("EMBEDDING_REGION")

                if not (effective_api_key or (project_id and region)):
                    return web.json_response({"error": "GCP credentials missing from env."}, status=400)

                try:
                    if effective_api_key:
                        cl = genai.Client(api_key=effective_api_key)
                    else:
                        cl = genai.Client(vertexai=True, project=project_id, location=region)

                    emb_res = cl.models.embed_content(
                        model="gemini-embedding-2-preview",
                        contents=query,
                    )
                    q_emb = emb_res.embeddings[0].values
                    results = search_similar_assets(q_emb, top_k=3, storage_mode=storage_mode, project_id=project_id)
                except Exception as e:
                    logger.error(f"Semantic Search API Error: {e}")
                    return web.json_response({"error": str(e)}, status=500)

            from urllib.parse import quote
            for item in results:
                path = item.get("filepath", "")
                if not path:
                    continue
                item["view_url"] = f"/google_genmedia/asset_manager/view?filepath={quote(path, safe='')}"

            return web.json_response({"results": results})

        @routes.post("/google_genmedia/asset_manager/copy_to_input")
        async def copy_to_input(request):
            import shutil
            import folder_paths
            data = await request.json()
            src_path = data.get("filepath", "")

            if not src_path:
                return web.json_response({"error": "No filepath specified"}, status=400)

            input_dir = folder_paths.get_input_directory()

            try:
                if src_path.startswith("gs://"):
                    # Handle GCS download
                    from google.cloud import storage
                    path_without_scheme = src_path[5:]
                    bucket_name, blob_name = path_without_scheme.split("/", 1)
                    filename = os.path.basename(blob_name)
                    dest_path = os.path.join(input_dir, filename)

                    client = storage.Client()
                    bucket = client.bucket(bucket_name)
                    blob = bucket.blob(blob_name)
                    blob.download_to_filename(dest_path)
                    return web.json_response({"filename": filename})
                else:
                    # Local File
                    if not os.path.exists(src_path):
                        return web.json_response({"error": f"File not found: {src_path}"}, status=404)
                    filename = os.path.basename(src_path)
                    dest_path = os.path.join(input_dir, filename)
                    shutil.copy2(src_path, dest_path)
                    return web.json_response({"filename": filename})
            except Exception as e:
                logger.error(f"Copy to input failed: {e}")
                return web.json_response({"error": str(e)}, status=500)

except ImportError as e:
    logger.warning(f"Could not initialize asset API routes inside ComfyUI: {e}")


NODE_CLASS_MAPPINGS = {
    "GeminiAssetIndexer": GeminiAssetIndexer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiAssetIndexer": "Asset Indexer (Gemini)",
}


class GeminiAssetIndexerWithPrompt(VertexAIClient):
    """
    Indexes an asset with a given Prompt, avoiding generation of extra captions.
    """
    def __init__(
        self,
        gcp_project_id: Optional[str] = None,
        gcp_region: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(
            gcp_project_id=gcp_project_id,
            gcp_region=gcp_region,
            user_agent=GEMINI_USER_AGENT,
            api_key=api_key,
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "A photo of a mountain", "multiline": True}),
                "custom_tags": ("STRING", {"default": "genai", "multiline": False}),
            },
            "optional": {
                "storage_mode": (["local", "gcs_bq"], {"default": "local"}),
                "gcs_bucket": ("STRING", {"default": ""}),
                "bq_dataset": ("STRING", {"default": "comfyui_assets"}),
                "bq_table": ("STRING", {"default": "media_index"}),
                "gcp_project_id": ("STRING", {"default": ""}),
                "gcp_region": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "asset_path")
    FUNCTION = "index_prompt_asset"
    CATEGORY = "Google AI/Asset Management"

    @api_error_retry
    def index_prompt_asset(
        self,
        image: torch.Tensor,
        prompt: str,
        custom_tags: str,
        storage_mode: str = "local",
        gcs_bucket: str = "",
        bq_dataset: str = "comfyui_assets",
        bq_table: str = "media_index",
        gcp_project_id: str = "",
        gcp_region: str = "",
        api_key: str = "",
    ):
        if gcp_project_id or gcp_region or api_key:
            self.__init__(
                gcp_project_id=gcp_project_id or None,
                gcp_region=gcp_region or None,
                api_key=api_key or None,
            )

        logger.info("Indexing asset mapped directly to static Prompt...")

        png_bytes = tensor_to_pil_to_bytes(image, format="PNG")
        timestamp = int(time.time())
        rand_id = random.randint(1000, 9999)
        filename = f"asset_{timestamp}_{rand_id}.png"
        filepath = os.path.join(MEDIA_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(png_bytes)

        # 1. Use Prompt as caption directly. Generate only tags with Pro.
        pro_model = GeminiModel.GEMINI_3_1_PRO.value
        tag_prompt = (
            f"Based on this scene and the description '{prompt}', return ONLY a valid JSON object with "
            "'tags' (a comma-separated string of 5 descriptive keywords)."
        )
        
        image_part = types.Part.from_bytes(data=png_bytes, mime_type="image/png")
        
        try:
            pro_res = self.client.models.generate_content(
                model=pro_model,
                contents=[tag_prompt, image_part],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                )
            )
            import json
            res_data = json.loads(pro_res.text.strip())
            res_tags = res_data.get("tags", "").strip()
            tags = f"{custom_tags}, {res_tags}"
        except Exception as e:
            logger.error(f"Tag extraction failed: {e}")
            tags = custom_tags

        # 2. Embed the content
        emb_model = GeminiEmbeddingModel.GEMINI_EMBEDDING_2.value
        embedding_list = None

        effective_api_key = os.environ.get("GEMINI_API_KEY")
        project_id = os.environ.get("GCP_PROJECT_ID")
        region = os.environ.get("EMBEDDING_REGION")

        if not (effective_api_key or (project_id and region)):
            return web.json_response({"error": "GCP credentials missing from env. Run an indexer node first to populate config."}, status=400)

        try:
            if effective_api_key:
                cl = genai.Client(api_key=effective_api_key)
            else:
                cl = genai.Client(vertexai=True, project=project_id, location=region)

            emb_res = cl.models.embed_content(
                model=emb_model,
                contents=image_part,
            )
            if emb_res.embeddings:
                embedding_list = emb_res.embeddings[0].values
        except Exception as e:
            logger.error(f"Embedding failed: {e}")

        # 3. Save
        caption = prompt
        try:
            if storage_mode == "gcs_bq":
                from .asset_db import insert_asset_cloud
                insert_asset_cloud(
                    file_bytes=png_bytes,
                    filename=filename,
                    filetype="image/png",
                    tags=[t.strip() for t in tags.split(",") if t.strip()],
                    caption=caption,
                    embedding_list=embedding_list,
                    gcs_bucket=gcs_bucket,
                    bq_dataset=bq_dataset,
                    bq_table=bq_table,
                    project_id=self.project_id if hasattr(self, "project_id") else gcp_project_id,
                )
            else:
                insert_or_update_asset(
                    filepath=filepath,
                    filetype="image/png",
                    tags=tags,
                    caption=caption,
                    embedding_list=embedding_list,
                )
        except Exception as e:
            logger.error(f"Prompt index save error: {e}")

        return (image, filepath)

NODE_CLASS_MAPPINGS["GeminiAssetIndexerWithPrompt"] = GeminiAssetIndexerWithPrompt
NODE_DISPLAY_NAME_MAPPINGS["GeminiAssetIndexerWithPrompt"] = "Asset Indexer With Prompt (Gemini)"


class GeminiVideoAssetIndexerWithPrompt(VertexAIClient):
    """
    Indexes a video asset paired to descriptive Prompt inputs.
    """
    def __init__(
        self,
        gcp_project_id: Optional[str] = None,
        gcp_region: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(
            gcp_project_id=gcp_project_id,
            gcp_region=gcp_region,
            user_agent=GEMINI_USER_AGENT,
            api_key=api_key,
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("VEO_VIDEO",),
                "prompt": ("STRING", {"default": "A video of moving clouds", "multiline": True}),
                "custom_tags": ("STRING", {"default": "video, visual", "multiline": False}),
            },
            "optional": {
                "storage_mode": (["local", "gcs_bq"], {"default": "local"}),
                "gcs_bucket": ("STRING", {"default": ""}),
                "bq_dataset": ("STRING", {"default": "comfyui_assets"}),
                "bq_table": ("STRING", {"default": "media_index"}),
                "gcp_project_id": ("STRING", {"default": ""}),
                "gcp_region": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("VEO_VIDEO", "STRING")
    RETURN_NAMES = ("video_path", "asset_path")
    FUNCTION = "index_video_prompt"
    CATEGORY = "Google AI/Asset Management"

    @api_error_retry
    def index_video_prompt(
        self,
        video_path,
        prompt: str,
        custom_tags: str,
        storage_mode: str = "local",
        gcs_bucket: str = "",
        bq_dataset: str = "comfyui_assets",
        bq_table: str = "media_index",
        gcp_project_id: str = "",
        gcp_region: str = "",
        api_key: str = "",
    ):
        if gcp_project_id or gcp_region or api_key:
            self.__init__(
                gcp_project_id=gcp_project_id or None,
                gcp_region=gcp_region or None,
                api_key=api_key or None,
            )

        # Get raw file string from ComfyUI video outputs
        target_file = ""
        if isinstance(video_path, list) and len(video_path) > 0:
            target_file = video_path[0]
        elif isinstance(video_path, str):
            target_file = video_path

        if not target_file or not os.path.exists(target_file):
            logger.error(f"Video indexing failed. Path not found: {target_file}")
            return (video_path, "")

        with open(target_file, "rb") as f:
            media_bytes = f.read()

        # Save local copy to asset_manager/media
        timestamp = int(time.time())
        rand_id = random.randint(1000, 9999)
        filename = f"video_{timestamp}_{rand_id}.mp4"
        local_filepath = os.path.join(MEDIA_DIR, filename)

        with open(local_filepath, "wb") as f:
            f.write(media_bytes)

        # Assign updated target_file to local_filepath for storage mappings
        target_file = local_filepath

        video_part = types.Part.from_bytes(data=media_bytes, mime_type="video/mp4")

        # 1. Tags
        pro_model = GeminiModel.GEMINI_3_1_PRO.value
        tag_prompt = (
            f"Describe the atmosphere and visual style for video query '{prompt}'. Return ONLY a valid JSON object "
            "with a property 'tags' containing 5 descriptive keywords separated by commas."
        )
        try:
            res = self.client.models.generate_content(
                model=pro_model,
                contents=[tag_prompt, video_part],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                )
            )
            import json
            res_data = json.loads(res.text.strip())
            tags = f"{custom_tags}, {res_data.get('tags', '')}"
        except Exception as e:
            logger.error(f"Tag generation error: {e}")
            tags = custom_tags

        # 2. Embedding
        emb_model = GeminiEmbeddingModel.GEMINI_EMBEDDING_2.value
        embedding_list = None

        effective_api_key = os.environ.get("GEMINI_API_KEY")
        project_id = os.environ.get("GCP_PROJECT_ID")
        region = os.environ.get("EMBEDDING_REGION")

        if not (effective_api_key or (project_id and region)):
            return web.json_response({"error": "GCP credentials missing from env. Run an indexer node first to populate config."}, status=400)

        try:
            if effective_api_key:
                cl = genai.Client(api_key=effective_api_key)
            else:
                cl = genai.Client(vertexai=True, project=project_id, location=region)

            emb_res = cl.models.embed_content(
                model=emb_model,
                contents=video_part,
            )
            if emb_res.embeddings:
                embedding_list = emb_res.embeddings[0].values
        except Exception as e:
            logger.error(f"Video embedding error: {e}")

        # 3. Store
        try:
            filename = os.path.basename(target_file)
            if storage_mode == "gcs_bq":
                from .asset_db import insert_asset_cloud
                insert_asset_cloud(
                    file_bytes=media_bytes,
                    filename=filename,
                    filetype="video/mp4",
                    tags=[t.strip() for t in tags.split(",") if t.strip()],
                    caption=prompt,
                    embedding_list=embedding_list,
                    gcs_bucket=gcs_bucket,
                    bq_dataset=bq_dataset,
                    bq_table=bq_table,
                    project_id=self.project_id if hasattr(self, "project_id") else gcp_project_id,
                )
            else:
                insert_or_update_asset(
                    filepath=target_file,
                    filetype="video/mp4",
                    tags=tags,
                    caption=prompt,
                    embedding_list=embedding_list,
                )
        except Exception as e:
            logger.error(f"Video storage failure: {e}")

        return (video_path, target_file)

NODE_CLASS_MAPPINGS["GeminiVideoAssetIndexerWithPrompt"] = GeminiVideoAssetIndexerWithPrompt
NODE_DISPLAY_NAME_MAPPINGS["GeminiVideoAssetIndexerWithPrompt"] = "Video Asset Indexer With Prompt (Gemini)"

