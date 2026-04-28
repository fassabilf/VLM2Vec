from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)


class SiglipModel(nn.Module):
    """
    SigLIP wrapper for use as a baseline backbone in VLM2Vec / MMEB evaluation.

    SigLIP is a dual-encoder model with separate text and image encoders.
    This wrapper provides a unified interface compatible with MMEBModel.encode_input().
    """

    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        processor: Optional[AutoProcessor] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        model_name = model_path or model_name

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)

        if processor is not None:
            self.processor = processor
        else:
            self.processor = AutoProcessor.from_pretrained(model_name)

        self.normalize = True
        self.text_hidden_size = self.model.config.text_config.hidden_size
        self.vision_hidden_size = self.model.config.vision_config.hidden_size
        logger.info(f"SigLIP model loaded: {model_name}, device: {self.device}")

    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of text strings into embeddings."""
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        outputs = self.model.get_text_features(**inputs)
        # get_text_features returns a tensor directly for SigLIP
        if hasattr(outputs, "pooler_output"):
            emb = outputs.pooler_output
        elif isinstance(outputs, tuple):
            emb = outputs[0]
        else:
            emb = outputs
        return emb

    def _encode_images(self, images: list) -> torch.Tensor:
        """Encode a list of PIL images into embeddings."""
        # Ensure all images are RGB PIL images
        pil_images = []
        for img in images:
            # Unwrap single-element list (from eval collator)
            if isinstance(img, list):
                img = img[0] if img else None
            if img is None:
                continue
            if not hasattr(img, 'mode'):
                continue
            pil_images.append(img.convert("RGB") if img.mode != "RGB" else img)

        if not pil_images:
            return torch.zeros(0, self.vision_hidden_size, device=self.device, dtype=torch.bfloat16)

        inputs = self.processor(
            images=pil_images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        outputs = self.model.get_image_features(**inputs)
        if hasattr(outputs, "pooler_output"):
            emb = outputs.pooler_output
        elif isinstance(outputs, tuple):
            emb = outputs[0]
        else:
            emb = outputs
        return emb

    def get_fused_embeddings(
        self,
        texts: List[str] = None,
        images: List[Optional[Image.Image]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Main interface called by MMEBModel.encode_input().

        Handles three cases:
        1. Text-only: encode text with text encoder
        2. Image-only: encode images with image encoder
        3. Fused (text + image): sum of text and image embeddings

        Args:
            texts: list of text strings (can contain None for image-only items)
            images: list of PIL images (can contain None for text-only items)

        Returns:
            Tensor of shape (batch_size, embed_dim)
        """
        batch_size = len(texts) if texts is not None else len(images)

        # Normalize images: unwrap single-element lists, extract PIL image
        def _unwrap_image(img):
            """Unwrap a potential list wrapper around an image."""
            if isinstance(img, list):
                # flatten: take first non-None element
                for i in img:
                    if i is not None:
                        return i
                return None
            return img

        if images is not None:
            images = [_unwrap_image(img) for img in images]

        # Determine which items have text and/or images
        has_text = [False] * batch_size
        has_image = [False] * batch_size

        if texts is not None:
            for i, t in enumerate(texts):
                if t is not None and isinstance(t, str) and len(t.strip()) > 0:
                    has_text[i] = True

        if images is not None:
            for i, img in enumerate(images):
                if img is not None:
                    has_image[i] = True

        embed_dim = self.text_hidden_size  # text and image share same dim in SigLIP
        result = torch.zeros(batch_size, embed_dim, device=self.device, dtype=torch.bfloat16)

        # Encode valid texts
        text_indices = [i for i in range(batch_size) if has_text[i]]
        if text_indices:
            valid_texts = [texts[i] for i in text_indices]
            with torch.no_grad():
                text_emb = self._encode_text(valid_texts)
            for idx, ti in enumerate(text_indices):
                result[ti] += text_emb[idx]

        # Encode valid images
        image_indices = [i for i in range(batch_size) if has_image[i]]
        if image_indices:
            valid_images = [images[i] for i in image_indices]
            with torch.no_grad():
                image_emb = self._encode_images(valid_images)
            for idx, ii in enumerate(image_indices):
                result[ii] += image_emb[idx]

        # Normalize
        if self.normalize:
            result = torch.nn.functional.normalize(result, p=2, dim=-1)

        return result.contiguous()


def custom_collate_fn(batch):
    return batch


if __name__ == "__main__":
    model = SiglipModel("google/siglip-so400m-patch14-384")

    # Text-only
    texts = ["A photo of a cat", "A photo of a dog"]
    e_text = model.get_fused_embeddings(texts=texts, images=[None, None])
    print("Text embeddings shape:", e_text.shape)

    # Image-only
    img = Image.new("RGB", (384, 384), color="red")
    e_img = model.get_fused_embeddings(texts=[None], images=[img])
    print("Image embeddings shape:", e_img.shape)

    # Fused
    e_fused = model.get_fused_embeddings(texts=["A cat"], images=[img])
    print("Fused embeddings shape:", e_fused.shape)
    print("Text-Image similarity:", (e_text[0] * e_img[0]).sum().item())
