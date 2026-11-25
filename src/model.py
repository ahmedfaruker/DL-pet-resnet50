"""ResNet-50 with a new 37-class head (transfer learning) + checkpoint I/O."""
import torch
import torch.nn as nn
from torchvision import models

from src import config as C


def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features          # 2048
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, C.NUM_CLASSES),  # raw logits; softmax applied at inference
    )
    return model


def param_groups(model: nn.Module):
    """Different learning rates: backbone (pretrained) vs head (new)."""
    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return [
        {"params": backbone, "lr": C.LR_BACKBONE},
        {"params": head, "lr": C.LR_HEAD},
    ]


def save_checkpoint(model, classes, path=C.CKPT_PATH, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes, "extra": extra or {}}, path)


def load_checkpoint(path=C.CKPT_PATH, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = build_model(pretrained=False)       # weights come from the checkpoint
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["classes"]