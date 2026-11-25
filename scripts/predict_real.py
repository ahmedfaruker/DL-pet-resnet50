"""Run the model on your own phone photos (assignment section 8).

Put photos in real_photos/ and fill real_photos/labels.csv:
    filename,true_label,true_species
    kedi1.jpg,British Shorthair,cat
    sokak_kedisi.jpg,unknown,cat
true_label: one of the 37 class names (see outputs/metrics/classes.txt), a breed
outside the dataset, or 'unknown' for mixed breeds.
true_species: 'cat' or 'dog'.
Run from project root:  python -m scripts.predict_real
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps

from src import config as C
from src.data import class_species, get_transforms
from src.model import load_checkpoint
from src.utils import get_device



def load_image(path):
    img = Image.open(path)
    return ImageOps.exif_transpose(img).convert("RGB")   # phone photos carry EXIF rotation


@torch.no_grad()
def predict(model, img, classes, device, k=3):
    x = get_transforms(train=False)(img).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), 1)[0]
    top = probs.topk(k)
    return [(classes[i], p.item()) for p, i in zip(top.values, top.indices)]


def main():
    device = get_device()
    model, classes = load_checkpoint(device=device)
    (C.METRIC_DIR / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
    species_by_name = {classes[i]: s for i, s in class_species().items()}

    with open(C.REAL_DIR / "labels.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = []
    for row in rows:
        img = load_image(C.REAL_DIR / row["filename"])
        top3 = predict(model, img, classes, device)
        true = row["true_label"].strip()
        results.append({
            "filename": row["filename"], "true_label": true,
            "pred_label": top3[0][0], "confidence": round(top3[0][1], 4),
            "correct": top3[0][0].lower() == true.lower(),
            "in_top3": any(c.lower() == true.lower() for c, _ in top3),
            "species_correct": row["true_species"].strip().lower() == species_by_name[top3[0][0]],
            "top3": "; ".join(f"{c} ({p:.2f})" for c, p in top3),
            "_img": img,
        })

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    axes = [axes] if n == 1 else axes
    for ax, r in zip(axes, results):
        ax.imshow(r["_img"]); ax.axis("off")
        # green: breed correct, orange: only species correct, red: species wrong
        color = "green" if r["correct"] else ("orange" if r["species_correct"] else "red")
        ax.set_title(f"True: {r['true_label']}\nPred: {r['pred_label']} ({r['confidence']:.0%})",
                     color=color, fontsize=10)
        ax.text(0.5, -0.05, r["top3"].replace("; ", "\n"), transform=ax.transAxes,
                ha="center", va="top", fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "real_photos_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    keys = [k for k in results[0] if not k.startswith("_")]
    with open(C.METRIC_DIR / "real_photos_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: r[k] for k in keys} for r in results)

    correct = sum(r["correct"] for r in results)
    species_ok = sum(r["species_correct"] for r in results)
    print(f"Real photos top-1 accuracy: {correct}/{n}")
    print(f"Species (cat/dog) accuracy: {species_ok}/{n}")
    for r in results:
        print(f"  {r['filename']}: true={r['true_label']} | {r['top3']}")


if __name__ == "__main__":
    main()