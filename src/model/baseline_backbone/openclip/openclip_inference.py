from __future__ import annotations

import logging
import os
from typing import List, Optional

import open_clip
import torch
from PIL import Image
from torch import nn

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


class OpenCLIPModel(nn.Module):
    """
    OpenCLIP wrapper for use as a baseline backbone in VLM2Vec / MMEB evaluation.

    OpenCLIP is a dual-encoder model with separate text and image encoders.
    This wrapper provides a unified interface compatible with MMEBModel.encode_input()
    via get_fused_embeddings(), mirroring the SigLIP / MetaCLIP2 baseline pattern.

    Supported models (examples):
      - apple/DFN2B-CLIP-ViT-B-16
      - laion/CLIP-ViT-g-14-laion2B-s12B-b42K
      - Any model available via open_clip on HF Hub
    """

    def __init__(
        self,
        model_name: str = "apple/DFN2B-CLIP-ViT-B-16",
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ) -> None:
        super().__init__()
        self.device = device

        is_local = model_path is not None and (
            os.path.isfile(model_path)
            or model_path.endswith((".pt", ".pth", ".safetensors"))
        )

        if is_local:
            # Local .pt checkpoint: model_name = arch (e.g. "ViT-T-16"),
            # model_path = path to .pt file
            logger.info(f"Loading local checkpoint: arch={model_name!r}, path={model_path!r}")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=model_path
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
        else:
            # HuggingFace Hub model
            hf_name = model_path or model_name
            logger.info(f"Loading HF Hub model: {hf_name!r}")
            self.model, self.preprocess = open_clip.create_model_from_pretrained(
                f"hf-hub:{hf_name}"
            )
            from transformers import AutoTokenizer
            local_cache = snapshot_download(repo_id=hf_name, local_files_only=True)
            try:
                self.tokenizer = open_clip.get_tokenizer(f"hf-hub:{hf_name}")
            except Exception:
                logger.warning("Falling back to AutoTokenizer from local cache...")
                hf_tok = AutoTokenizer.from_pretrained(local_cache)
                context_len = self.model.context_length
                class _TokenizerWrapper:
                    def __call__(self_, texts, context_length=None):
                        cl = context_length or context_len
                        return hf_tok(texts, return_tensors="pt", padding="max_length",
                                    truncation=True, max_length=cl)["input_ids"]
                self.tokenizer = _TokenizerWrapper()

        self.model = self.model.to(self.device)
        self.model.eval()

        self.normalize = True
        try:
            embed_dim = self.model.text_projection.shape[1]
        except AttributeError:
            logger.warning("text_projection not found, inferring embed_dim from text encoder...")
            try:
                embed_dim = self.model.text.output_dim
            except AttributeError:
                embed_dim = self.model.text.config.hidden_size

        self.text_hidden_size = embed_dim
        self.vision_hidden_size = embed_dim
        
        logger.info(
            f"OpenCLIP model loaded: {model_name}, device: {self.device}, embed_dim: {embed_dim}"
        )

    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of text strings into embeddings."""
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens)
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
            return torch.zeros(
                0, self.vision_hidden_size, device=self.device, dtype=torch.bfloat16
            )

        image_tensor = torch.stack([self.preprocess(img) for img in pil_images]).to(
            self.device
        )
        with torch.no_grad():
            emb = self.model.encode_image(image_tensor)
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

        embed_dim = self.text_hidden_size
        result = torch.zeros(
            batch_size, embed_dim, device=self.device, dtype=torch.bfloat16
        )

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


def custom_collate_fn(batch):
    return batch


if __name__ == "__main__":
    model = OpenCLIPModel("apple/DFN2B-CLIP-ViT-B-16")

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