"""Plain PyTorch loader for the CSD checkpoint (answers #8: "How to load the model
as a PyTorch module?").

The released `pytorch_model.bin` has no `transformers` AutoModel support and the repo's
`config.json` carries no architecture info ({"model_type": "custom"}), so this module
reconstructs the architecture directly from the checkpoint's state_dict key names and
shapes:

- The backbone is a stock CLIP ViT-L/14 (quickgelu) visual transformer -- quickgelu to
  match OpenAI's original CLIP activation exactly, since CSD's training started from
  OpenAI's CLIP weights, not open_clip's plain-GELU variant.
- The backbone's own contrastive `proj` is unused -- the checkpoint has no `proj` key,
  so the raw pooled feature (1024-dim) feeds forward instead.
- `last_layer_style` / `last_layer_content` are separate (1024, 768) matrices, applied
  via matrix multiply, not `nn.Linear` (the checkpoint stores them as raw (in, out)
  matrices, not the (out, in) shape `nn.Linear.weight` expects).

Loading with `strict=True` against this class produces zero missing and zero unexpected
keys against the released checkpoint.

Requires: torch, open_clip_torch, huggingface_hub, pillow, torchvision.
"""

from __future__ import annotations

import os
from pathlib import Path

import open_clip
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

CSD_REPO_ID = "tomg-group-umd/CSD-ViT-L"

# OpenAI's published CLIP normalization stats -- the backbone was trained on images
# preprocessed this way, so inference must match exactly.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

csd_preprocess = transforms.Compose(
    [
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
    ]
)


class CSD_CLIP(nn.Module):
    """CLIP ViT-L/14 backbone (no contrastive projection) + separate style/content heads."""

    def __init__(self, model_name: str = "ViT-L-14-quickgelu", feature_dim: int = 1024, embed_dim: int = 768):
        super().__init__()
        clip_model = open_clip.create_model(model_name, pretrained=False)
        self.backbone = clip_model.visual
        self.backbone.proj = None
        self.last_layer_style = nn.Parameter(torch.randn(feature_dim, embed_dim))
        self.last_layer_content = nn.Parameter(torch.randn(feature_dim, embed_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (pooled_feature, style_embed, content_embed)."""
        feat = self.backbone(x)
        style_embed = feat @ self.last_layer_style
        content_embed = feat @ self.last_layer_content
        return feat, style_embed, content_embed


def load_csd_model(cache_dir: str | Path | None = None, device: str | None = None) -> CSD_CLIP:
    """Download (if needed) and load the CSD checkpoint as a plain PyTorch module.

    Example:
        model = load_csd_model()
        model.eval()
        with torch.no_grad():
            img = csd_preprocess(Image.open("photo.jpg")).unsqueeze(0)
            _, style_embed, content_embed = model(img)
    """
    from huggingface_hub import snapshot_download

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    csd_path = snapshot_download(repo_id=CSD_REPO_ID, cache_dir=str(cache_dir) if cache_dir else None)

    checkpoint = torch.load(
        os.path.join(csd_path, "pytorch_model.bin"), map_location="cpu", weights_only=False
    )
    raw_sd = checkpoint["model_state_dict"]
    stripped_sd = {
        (k[len("module."):] if k.startswith("module.") else k): v for k, v in raw_sd.items()
    }

    model = CSD_CLIP()
    missing, unexpected = model.load_state_dict(stripped_sd, strict=False)
    assert not missing and not unexpected, (
        f"CSD checkpoint no longer matches CSD_CLIP -- missing={missing}, unexpected={unexpected}"
    )

    return model.to(device).eval()
