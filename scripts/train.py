"""Train ResNet-50 on Oxford-IIIT Pet.
Saves best checkpoint (by val accuracy), training summary and loss/accuracy curves.
Run from project root:  python -m scripts.train
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src import config as C
from src.data import get_loaders
from src.model import build_model, param_groups, save_checkpoint
from src.utils import device_name, dump_json, get_device, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """One pass over the loader. Trains if optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    use_amp = C.USE_AMP and device.type == "cuda"

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast(device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


def plot_history(h, path):
    ep = range(1, len(h["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, h["train_loss"], label="train")
    ax[0].plot(ep, h["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, h["train_acc"], label="train")
    ax[1].plot(ep, h["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    set_seed(C.SEED)
    device = get_device()
    print(f"Device: {device_name(device)}")

    train_loader, val_loader, _, classes = get_loaders()
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Classes: {len(classes)}")

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=C.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(param_groups(model), weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=C.EPOCHS)
    scaler = torch.amp.GradScaler(device.type, enabled=C.USE_AMP and device.type == "cuda")

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "epoch_time")}
    best_acc, best_epoch, bad_epochs = 0.0, 0, 0
    t_start = time.perf_counter()

    for epoch in range(1, C.EPOCHS + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        dt = time.perf_counter() - t0

        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc, dt)):
            history[k].append(v)
        print(f"[{epoch:02d}/{C.EPOCHS}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {dt:.1f}s")

        if va_acc > best_acc:
            best_acc, best_epoch, bad_epochs = va_acc, epoch, 0
            save_checkpoint(model, classes, extra={"epoch": epoch, "val_acc": va_acc})
        else:
            bad_epochs += 1
            if bad_epochs >= C.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    total_time = time.perf_counter() - t_start
    summary = {
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "epochs_run": len(history["train_loss"]),
        "total_train_time_sec": total_time,
        "device": device_name(device),
        "hyperparameters": {
            "img_size": C.IMG_SIZE, "batch_size": C.BATCH_SIZE, "epochs": C.EPOCHS,
            "lr_backbone": C.LR_BACKBONE, "lr_head": C.LR_HEAD,
            "weight_decay": C.WEIGHT_DECAY, "label_smoothing": C.LABEL_SMOOTHING,
            "optimizer": "AdamW", "scheduler": "CosineAnnealingLR", "amp": C.USE_AMP,
        },
        "history": history,
    }
    dump_json(summary, C.METRIC_DIR / "train_summary.json")
    plot_history(history, C.FIG_DIR / "training_curves.png")
    print(f"Done in {total_time / 60:.1f} min. Best val acc {best_acc:.4f} (epoch {best_epoch})")


if __name__ == "__main__":   # required for DataLoader workers on Windows
    main()