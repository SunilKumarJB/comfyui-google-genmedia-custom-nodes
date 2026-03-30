# Copyright 2025 Google LLC
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

# This is a preview version of Gemini 3 Pro Image custom node

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .constants import (
    GEMINI_3_PRO_IMAGE_ASPECT_RATIO,
    GeminiProImageModel,
    ThresholdOptions,
)
from .custom_exceptions import APIExecutionError, APIInputError, ConfigurationError
from .gemini_pro_image_api import GeminiProImageAPI


class Gemini3ProImage:
    """
    A ComfyUI node for generating images from text prompts using the Google Imagen API.
    """

    def __init__(self) -> None:
        """
        Initializes the ImagenTextToImageNode.
        """
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """
        Defines the input types and widgets for the ComfyUI node.

        Returns:
            A dictionary specifying the required and optional input parameters.
        """
        return {
            "required": {
                "model": (
                    [model.name for model in GeminiProImageModel],
                    {"default": GeminiProImageModel.GEMINI_3_PRO_IMAGE.name},
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A vivid landscape painting of a futuristic city",
                    },
                ),
                "aspect_ratio": (
                    [
                        "1:1",
                        "2:3",
                        "3:2",
                        "3:4",
                        "4:3",
                        "4:5",
                        "5:4",
                        "9:16",
                        "16:9",
                        "21:9",
                    ],
                    {"default": "16:9"},
                ),
                "image_size": (
                    [
                        "1K",
                        "2K",
                        "4K",
                    ],
                    {"default": "1K"},
                ),
                "output_mime_type": (
                    [
                        "PNG",
                        "JPEG",
                    ],
                    {"default": "PNG"},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "top_k": ("INT", {"default": 32, "min": 1, "max": 64}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                # Safety Settings
                "harassment_threshold": (
                    [threshold_option.name for threshold_option in ThresholdOptions],
                    {"default": ThresholdOptions.BLOCK_MEDIUM_AND_ABOVE.name},
                ),
                "hate_speech_threshold": (
                    [threshold_option.name for threshold_option in ThresholdOptions],
                    {"default": ThresholdOptions.BLOCK_MEDIUM_AND_ABOVE.name},
                ),
                "sexually_explicit_threshold": (
                    [threshold_option.name for threshold_option in ThresholdOptions],
                    {"default": ThresholdOptions.BLOCK_MEDIUM_AND_ABOVE.name},
                ),
                "dangerous_content_threshold": (
                    [threshold_option.name for threshold_option in ThresholdOptions],
                    {"default": ThresholdOptions.BLOCK_MEDIUM_AND_ABOVE.name},
                ),
                "system_instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Optional system instruction for the model",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Google GenAI API Key (prioritized over environment variable)",
                    },
                ),
                "gcp_project_id": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "GCP project id where Vertex AI API will query Gemini Pro Image",
                    },
                ),
                "gcp_region": (
                    "STRING",
                    {
                        "default": "global",
                        "tooltip": "GCP region for Vertex AI API",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Generated Image",)

    FUNCTION = "generate_and_return_image"
    CATEGORY = "Google AI/GeminiProImage"

    def generate_and_return_image(
        self,
        model: str,
        aspect_ratio: str,
        image_size: str,
        output_mime_type: str,
        prompt: str,
        temperature: float,
        top_p: float,
        top_k: int,
        hate_speech_threshold: str,
        harassment_threshold: str,
        sexually_explicit_threshold: str,
        dangerous_content_threshold: str,
        system_instruction: str,
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
        image3: Optional[torch.Tensor] = None,
        image4: Optional[torch.Tensor] = None,
        image5: Optional[torch.Tensor] = None,
        image6: Optional[torch.Tensor] = None,
        api_key: str = "",
        gcp_project_id: Optional[str] = None,
        gcp_region: Optional[str] = None,
    ) -> Tuple[torch.Tensor,]:
        """Generates images using the Gemini Pro Image API and returns them.

        This method interfaces with the GeminiProImageAPI to generate images
        based on a prompt and other parameters. It then converts the generated
        PIL images into a PyTorch tensor suitable for use in ComfyUI.

        Args:
            model: The Gemini Pro Image model to use. default: gemini-3-pro-image-preview
            aspect_ratio: The desired aspect ratio of the output image.
            image_size: The desired image size for the output image.
            output_mime_type: The desired format for the output image.
            prompt: The text prompt for image generation.
            temperature: Controls randomness in token generation.
            top_p: The cumulative probability of tokens to consider for sampling.
            top_k: The number of highest probability tokens to consider for sampling.
            hate_speech_threshold: Safety threshold for hate speech.
            harassment_threshold: Safety threshold for harassment.
            sexually_explicit_threshold: Safety threshold for sexually explicit
              content.
            dangerous_content_threshold: Safety threshold for dangerous content.
            system_instruction: System-level instructions for the model.
            image1: The primary input image tensor for image editing tasks.
            image2: An optional second input image tensor. Defaults to None.
            image3: An optional third input image tensor. Defaults to None.
            image4: An optional fourth input image tensor. Defaults to None.
            image5: An optional fifth input image tensor. Defaults to None.
            image6: An optional sixth input image tensor. Defaults to None.
            api_key: Google GenAI API Key.
            gcp_project_id: The GCP project ID.
            gcp_region: The GCP region.

        Returns:
            A tuple containing a PyTorch tensor of the generated images,
            formatted as (batch_size, height, width, channels).

        Raises:
            RuntimeError: If API configuration fails, or if image generation encounters an API error.
        """
        try:
            init_api_key = api_key if api_key else None
            gemini_pro_image_api = GeminiProImageAPI(
                project_id=gcp_project_id, region=gcp_region, api_key=init_api_key
            )
        except ConfigurationError as e:
            raise RuntimeError(
                f"Gemini Pro Image API Configuration Error: {e}"
            ) from e

        output_mime_type = "image/" + output_mime_type.lower()

        if aspect_ratio not in GEMINI_3_PRO_IMAGE_ASPECT_RATIO:
            raise RuntimeError(
                f"Invalid aspect ratio: {aspect_ratio}. Valid aspect ratios are: {GEMINI_3_PRO_IMAGE_ASPECT_RATIO}."
            )

        try:
            pil_images = gemini_pro_image_api.generate_image(
                model=model,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                output_mime_type=output_mime_type,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                hate_speech_threshold=hate_speech_threshold,
                harassment_threshold=harassment_threshold,
                sexually_explicit_threshold=sexually_explicit_threshold,
                dangerous_content_threshold=dangerous_content_threshold,
                system_instruction=system_instruction,
                image1=image1,
                image2=image2,
                image3=image3,
                image4=image4,
                image5=image5,
                image6=image6,
            )
        except APIInputError as e:
            raise RuntimeError(f"Image generation input error: {e}") from e
        except APIExecutionError as e:
            raise RuntimeError(f"Image generation API error: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred during image generation: {e}"
            ) from e

        if not pil_images:
            raise RuntimeError(
                "Gemini Pro Image API failed to generate images or generated no valid images."
            )

        output_tensors: List[torch.Tensor] = []
        for img in pil_images:
            img = img.convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)[None,]
            output_tensors.append(img_tensor)

        batched_images_tensor = torch.cat(output_tensors, dim=0)
        return (batched_images_tensor,)


class Gemini3ProImageEditing:
    """
    A ComfyUI node for editing images (inpainting, outpainting) using Gemini 3 Pro.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "model": (
                    [model.name for model in GeminiProImageModel],
                    {"default": GeminiProImageModel.GEMINI_3_PRO_IMAGE.name},
                ),
                "prompt": ("STRING", {"multiline": True, "default": "Describe the edits"}),
                "base_image": ("IMAGE",),
                "edit_mode": (["INPAINT", "OUTPAINT", "EDIT"], {"default": "INPAINT"}),
            },
            "optional": {
                "mask": ("MASK",),
                "output_mime_type": (["PNG", "JPEG"], {"default": "PNG"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 32, "min": 1, "max": 64}),
                "harassment_threshold": (
                    [threshold_option.name for threshold_option in ThresholdOptions],
                    {"default": ThresholdOptions.BLOCK_MEDIUM_AND_ABOVE.name},
                ),
                "hate_speech_threshold": (
                    [threshold_option.name for threshold_option in ThresholdOptions],
                    {"default": ThresholdOptions.BLOCK_MEDIUM_AND_ABOVE.name},
                ),
                "sexually_explicit_threshold": (
                    [threshold_option.name for threshold_option in ThresholdOptions],
                    {"default": ThresholdOptions.BLOCK_MEDIUM_AND_ABOVE.name},
                ),
                "dangerous_content_threshold": (
                    [threshold_option.name for threshold_option in ThresholdOptions],
                    {"default": ThresholdOptions.BLOCK_MEDIUM_AND_ABOVE.name},
                ),
                "system_instruction": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "gcp_project_id": ("STRING", {"default": ""}),
                "gcp_region": ("STRING", {"default": "global"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Edited Image",)
    FUNCTION = "edit"
    CATEGORY = "Google AI/GeminiProImage"

    def edit(
        self,
        model: str,
        prompt: str,
        base_image: torch.Tensor,
        edit_mode: str,
        mask: Optional[torch.Tensor] = None,
        output_mime_type: str = "PNG",
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 32,
        hate_speech_threshold: str = "BLOCK_MEDIUM_AND_ABOVE",
        harassment_threshold: str = "BLOCK_MEDIUM_AND_ABOVE",
        sexually_explicit_threshold: str = "BLOCK_MEDIUM_AND_ABOVE",
        dangerous_content_threshold: str = "BLOCK_MEDIUM_AND_ABOVE",
        system_instruction: str = "",
        api_key: str = "",
        gcp_project_id: Optional[str] = None,
        gcp_region: Optional[str] = None,
    ) -> Tuple[torch.Tensor,]:
        try:
            init_api_key = api_key if api_key else None
            api = GeminiProImageAPI(project_id=gcp_project_id, region=gcp_region, api_key=init_api_key)
        except ConfigurationError as e:
            raise RuntimeError(f"Gemini API Configuration Error: {e}") from e

        mime_type = "image/" + output_mime_type.lower()

        # Handle mask conversion if provided (ComfyUI masks are typically [B, H, W])
        mask_tensor = None
        if mask is not None:
            if len(mask.shape) == 3: # [B, H, W]
                mask_tensor = mask.unsqueeze(-1).repeat(1, 1, 1, 3) # [B, H, W, 3]
            else:
                mask_tensor = mask

        try:
            pil_images = api.edit_image(
                model=model,
                prompt=prompt,
                base_image=base_image,
                mask=mask_tensor,
                edit_mode=edit_mode,
                output_mime_type=mime_type,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                hate_speech_threshold=hate_speech_threshold,
                harassment_threshold=harassment_threshold,
                sexually_explicit_threshold=sexually_explicit_threshold,
                dangerous_content_threshold=dangerous_content_threshold,
                system_instruction=system_instruction,
            )
        except Exception as e:
            raise RuntimeError(f"Image editing failed: {e}") from e

        output_tensors: List[torch.Tensor] = []
        for img in pil_images:
            img = img.convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)[None,]
            output_tensors.append(img_tensor)

        return (torch.cat(output_tensors, dim=0),)


class Gemini3ProControlledImage:
    """
    A ComfyUI node for generating images with control signals (e.g. Canny) using Gemini 3 Pro.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "model": (
                    [model.name for model in GeminiProImageModel],
                    {"default": GeminiProImageModel.GEMINI_3_PRO_IMAGE.name},
                ),
                "prompt": ("STRING", {"multiline": True, "default": "A vivid landscape"}),
                "control_image": ("IMAGE",),
                "control_type": (["CANNY", "FACE_MESH"], {"default": "CANNY"}),
            },
            "optional": {
                "output_mime_type": (["PNG", "JPEG"], {"default": "PNG"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 32, "min": 1, "max": 64}),
                "harassment_threshold": (
                    [threshold_option.name for threshold_option in ThresholdOptions],
                    {"default": ThresholdOptions.BLOCK_MEDIUM_AND_ABOVE.name},
                ),
                "hate_speech_threshold": (
                    [threshold_option.name for threshold_option in ThresholdOptions],
                    {"default": ThresholdOptions.BLOCK_MEDIUM_AND_ABOVE.name},
                ),
                "sexually_explicit_threshold": (
                    [threshold_option.name for threshold_option in ThresholdOptions],
                    {"default": ThresholdOptions.BLOCK_MEDIUM_AND_ABOVE.name},
                ),
                "dangerous_content_threshold": (
                    [threshold_option.name for threshold_option in ThresholdOptions],
                    {"default": ThresholdOptions.BLOCK_MEDIUM_AND_ABOVE.name},
                ),
                "system_instruction": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "gcp_project_id": ("STRING", {"default": ""}),
                "gcp_region": ("STRING", {"default": "global"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Controlled Image",)
    FUNCTION = "generate"
    CATEGORY = "Google AI/GeminiProImage"

    def generate(
        self,
        model: str,
        prompt: str,
        control_image: torch.Tensor,
        control_type: str,
        output_mime_type: str = "PNG",
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 32,
        hate_speech_threshold: str = "BLOCK_MEDIUM_AND_ABOVE",
        harassment_threshold: str = "BLOCK_MEDIUM_AND_ABOVE",
        sexually_explicit_threshold: str = "BLOCK_MEDIUM_AND_ABOVE",
        dangerous_content_threshold: str = "BLOCK_MEDIUM_AND_ABOVE",
        system_instruction: str = "",
        api_key: str = "",
        gcp_project_id: Optional[str] = None,
        gcp_region: Optional[str] = None,
    ) -> Tuple[torch.Tensor,]:
        try:
            init_api_key = api_key if api_key else None
            api = GeminiProImageAPI(project_id=gcp_project_id, region=gcp_region, api_key=init_api_key)
        except ConfigurationError as e:
            raise RuntimeError(f"Gemini API Configuration Error: {e}") from e

        mime_type = "image/" + output_mime_type.lower()

        try:
            pil_images = api.generate_controlled_image(
                model=model,
                prompt=prompt,
                control_image=control_image,
                control_type=control_type,
                output_mime_type=mime_type,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                hate_speech_threshold=hate_speech_threshold,
                harassment_threshold=harassment_threshold,
                sexually_explicit_threshold=sexually_explicit_threshold,
                dangerous_content_threshold=dangerous_content_threshold,
                system_instruction=system_instruction,
            )
        except Exception as e:
            raise RuntimeError(f"Controlled image generation failed: {e}") from e

        output_tensors: List[torch.Tensor] = []
        for img in pil_images:
            img = img.convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)[None,]
            output_tensors.append(img_tensor)

        return (torch.cat(output_tensors, dim=0),)


NODE_CLASS_MAPPINGS = {
    "Gemini3ProImage": Gemini3ProImage,
    "Gemini3ProImageEditing": Gemini3ProImageEditing,
    "Gemini3ProControlledImage": Gemini3ProControlledImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini3ProImage": "Gemini 3 Pro Image",
    "Gemini3ProImageEditing": "Gemini 3 Pro Image Editing (Inpaint/Outpaint)",
    "Gemini3ProControlledImage": "Gemini 3 Pro Controlled Image (Canny/FaceMesh)",
}
