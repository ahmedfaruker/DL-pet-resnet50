"""Gradio web app: upload a cat/dog photo -> top-5 breed predictions.
Run from project root:  python -m app.gradio_app
"""
import time

import gradio as gr
import torch
from PIL import Image, ImageOps

from src.data import get_transforms
from src.model import load_checkpoint
from src.utils import get_device

DEVICE = get_device()
MODEL, CLASSES = load_checkpoint(device=DEVICE)
TF = get_transforms(train=False)
LOW_CONF = 0.5   # below this, warn that the animal may be a mixed / unseen breed


@torch.no_grad()
def classify(img: Image.Image):
    if img is None:
        return {}, ""
    img = ImageOps.exif_transpose(img).convert("RGB")
    x = TF(img).unsqueeze(0).to(DEVICE)
    t0 = time.perf_counter()
    probs = torch.softmax(MODEL(x), 1)[0]
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000

    top = probs.topk(5)
    result = {CLASSES[i]: float(p) for p, i in zip(top.values, top.indices)}
    info = f"Inference: {ms:.1f} ms on {DEVICE.type.upper()}"
    if top.values[0] < LOW_CONF:
        info += " | Low confidence: may be a mixed breed or a breed not in the dataset"
    return result, info


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Cat or dog photo"),
    outputs=[gr.Label(num_top_classes=5, label="Breed prediction"), gr.Textbox(label="Info")],
    title="Pet Breed Classifier (ResNet-50, Oxford-IIIT Pet)",
    description="Upload a photo of a cat or dog. The model predicts one of 37 breeds.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()