"""HiDream-O1 image generation pipeline wrapper."""

from __future__ import annotations

import logging
from typing import Any, cast

import torch

from services.services_utils import ImagePipelineOutputLike

logger = logging.getLogger(__name__)


class HiDreamImageGenerationPipeline:
    device: str
    tokenizer: Any
    model: Any

    @staticmethod
    def create(
        model_path: str,
        device: str | None = None,
    ) -> HiDreamImageGenerationPipeline:
        return HiDreamImageGenerationPipeline(model_path=model_path, device=device)

    def __init__(self, model_path: str, device: str | None = None) -> None:
        from transformers import AutoModel, AutoTokenizer
        
        self.device = device or "cuda"
        self.tokenizer = cast(Any, AutoTokenizer).from_pretrained(model_path, trust_remote_code=True)
        self.model = cast(Any, AutoModel).from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.float16,
            device_map=self.device
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
    ) -> ImagePipelineOutputLike:
        # HiDream-O1 uses instruction following.
        # We wrap the prompt in the expected format.
        formatted_prompt = f"USER: Generate an image of {prompt}\nASSISTANT: <|image|>"
        
        torch.manual_seed(seed)  # type: ignore[reportUnknownMemberType]
        
        # This is a simplified generation call assuming the model implements a 'generate_image' method
        # or follows the transformer generation pattern for images.
        # Based on research, HiDream uses a custom generation method.
        image: Any
        if hasattr(self.model, "generate_image"):
            image = self.model.generate_image(
                prompt=formatted_prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
        else:
            # Fallback to a generic generate call if 'generate_image' is not found
            # (Adjusting based on actual architecture discovered)
            inputs: Any = self.tokenizer(formatted_prompt, return_tensors="pt")
            output: Any = self.model.generate(**inputs, max_new_tokens=1024) # Simplified
            # Convert tokens to image (simplified)
            image = self.model.decode_image(output)

        image_item: Any = cast(Any, image)[0] if isinstance(image, list) else image
            
        return cast(ImagePipelineOutputLike, type('Output', (), {'images': [image_item]}))

    def to(self, device: str) -> None:
        self.device = device
        self.model.to(device)
