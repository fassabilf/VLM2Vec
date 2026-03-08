from __future__ import annotations

import logging
from typing import List, Optional

import torch
from PIL import Image
from torch import nn
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)


class MetaCLIP2Model(nn.Module):
    """
    MetaCLIP 2 wrapper for use as a baseline backbone in VLM2Vec / MMEB evaluation.

    MetaCLIP 2 is a multilingual vision-language dual-encoder model.
    This wrapper provides a unified interface compatible with MMEBModel.encode_input()
    via get_fused_embeddings(), mirroring the SigLIP baseline pattern.

    Supported models:
      - facebook/metaclip-2-worldwide-huge-quickgelu  (~2B, embed_dim=1024)
      - facebook/metaclip-2-mt5-worldwide-b32         (~254M, embed_dim=512)
    """

    def __init__(
        self,
        model_name: str = "facebook/metaclip-2-mt5-worldwide-b32",
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
        self.embed_dim = self.model.config.projection_dim
        logger.info(f"MetaCLIP2 model loaded: {model_name}, device: {self.device}, embed_dim: {self.embed_dim}")

    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of text strings into embeddings."""
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.get_text_features(**inputs)
        if hasattr(outputs, "pooler_output"):
            emb = outputs.pooler_output
        elif isinstance(outputs, tuple):
            emb = outputs[0]
        else:
            emb = outputs
        return emb.to(torch.bfloat16)

    def _encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode a list of PIL images into embeddings."""
        pil_images = []
        for img in images:
            # Unwrap single-element list (from eval collator)
            if isinstance(img, list):
                img = img[0] if img else None
            if img is None:
                continue
            if not hasattr(img, "mode"):
                continue
            pil_images.append(img.convert("RGB") if img.mode != "RGB" else img)

        if not pil_images:
            return torch.zeros(0, self.embed_dim, device=self.device, dtype=torch.bfloat16)

        inputs = self.processor(
            images=pil_images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.get_image_features(**inputs)
        if hasattr(outputs, "pooler_output"):
            emb = outputs.pooler_output
        elif isinstance(outputs, tuple):
            emb = outputs[0]
        else:
            emb = outputs
        return emb.to(torch.bfloat16)

    def get_fused_embeddings(
        self,
        texts: Optional[List[Optional[str]]] = None,
        images: Optional[List[Optional[Image.Image]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Main interface called by MMEBModel.encode_input().
        Handles three cases:
          1. Text-only: encode with text encoder
          2. Image-only: encode with image encoder
          3. Fused (text + image): sum of text and image embeddings, then L2-normalize

        Args:
            texts:  list of text strings; may contain None for image-only items
            images: list of PIL images; may contain None for text-only items

        Returns:
            Tensor of shape (batch_size, embed_dim)
        """
        batch_size = len(texts) if texts is not None else len(images)

        def _unwrap_image(img):
            if isinstance(img, list):
                for i in img:
                    if i is not None:
                        return i
                return None
            return img

        if images is not None:
            images = [_unwrap_image(img) for img in images]

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

        result = torch.zeros(batch_size, self.embed_dim, device=self.device, dtype=torch.bfloat16)

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

        if self.normalize:
            result = torch.nn.functional.normalize(result, p=2, dim=-1)

        return result.contiguous()


if __name__ == "__main__":
    model = MetaCLIP2Model("facebook/metaclip-2-mt5-worldwide-b32")

    # Text-only
    texts = ["A photo of a cat", "A photo of a dog"]
    e_text = model.get_fused_embeddings(texts=texts, images=[None, None])
    print("Text embeddings shape:", e_text.shape)

    # Image-only
    img = Image.new("RGB", (224, 224), color="red")
    e_img = model.get_fused_embeddings(texts=[None], images=[img])
    print("Image embeddings shape:", e_img.shape)

    # Fused
    e_fused = model.get_fused_embeddings(texts=["A red square"], images=[img])
    print("Fused embeddings shape:", e_fused.shape)
    print("Text-Image similarity:", (e_text[0] * e_img[0]).sum().item())
