"""Evaluate the best checkpoint on the official test split.
Outputs: metrics/test_metrics.json, metrics/classification_report.txt, figures/confusion_matrix.png
Run from project root:  python -m scripts.evaluate
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, top_k_accuracy_score)

from src import config as C
from src.data import get_loaders
from src.model import load_checkpoint
from src.utils import device_name, dump_json, get_device


@torch.no_grad()
def predict_all(model, loader, device):
    probs, labels = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), 1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def measure_latency(model, device, runs=100, warmup=10):
    """Single-image (batch=1) latency in milliseconds."""
    x = torch.randn(1, 3, C.IMG_SIZE, C.IMG_SIZE, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()       # wait for GPU, otherwise timing is wrong
    return (time.perf_counter() - t0) / runs * 1000


def plot_confusion(cm, classes, path):
    cm_norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Normalized confusion matrix (test)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    device = get_device()
    model, classes = load_checkpoint(device=device)
    _, _, test_loader, _ = get_loaders()

    t0 = time.perf_counter()
    probs, y_true = predict_all(model, test_loader, device)
    total_pred_time = time.perf_counter() - t0
    y_pred = probs.argmax(1)

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    metrics = {
        "test_samples": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "top5_accuracy": top_k_accuracy_score(y_true, probs, k=5, labels=range(len(classes))),
        "precision_macro": p, "recall_macro": r, "f1_macro": f1,
        "test_set_prediction_time_sec": total_pred_time,
        "latency_ms_per_image_batch1": measure_latency(model, device),
        "device": device_name(device),
    }
    if device.type == "cuda":   # also report CPU latency for comparison
        metrics["latency_ms_per_image_cpu"] = measure_latency(model.cpu(), torch.device("cpu"), runs=20)

    report = classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0)
    (C.METRIC_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    dump_json(metrics, C.METRIC_DIR / "test_metrics.json")

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, classes, C.FIG_DIR / "confusion_matrix.png")

    # most confused class pairs -> useful for the failure analysis section
    np.fill_diagonal(cm, 0)
    print("Most confused pairs (true -> pred : count):")
    for k in np.argsort(cm.ravel())[::-1][:5]:
        i, j = divmod(k, cm.shape[1])
        print(f"  {classes[i]} -> {classes[j]} : {cm[i, j]}")

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()