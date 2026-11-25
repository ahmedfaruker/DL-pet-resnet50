"""Dataset loading, transforms and train/val/test split."""
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src import config as C


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(C.IMG_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(C.MEAN, C.STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(C.IMG_SIZE * 1.14)),   # 256 for 224
        transforms.CenterCrop(C.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(C.MEAN, C.STD),
    ])


def _pet(split: str, train_tf: bool):
    return datasets.OxfordIIITPet(
        root=C.DATA_DIR, split=split, target_types="category",
        transform=get_transforms(train_tf), download=True,
    )


def _labels_of(ds):
    # torchvision keeps labels in a private list; fall back to iterating if it changes
    labels = getattr(ds, "_labels", None)
    if labels is None:
        labels = [ds[i][1] for i in range(len(ds))]
    return np.asarray(labels)


def get_datasets():
    """Official 'trainval' -> stratified train/val, official 'test' -> test."""
    trainval_aug = _pet("trainval", train_tf=True)
    trainval_plain = _pet("trainval", train_tf=False)
    test_ds = _pet("test", train_tf=False)

    labels = _labels_of(trainval_plain)
    idx = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        idx, test_size=C.VAL_RATIO, stratify=labels, random_state=C.SEED
    )
    train_ds = Subset(trainval_aug, train_idx)
    val_ds = Subset(trainval_plain, val_idx)
    return train_ds, val_ds, test_ds, test_ds.classes


def get_loaders():
    train_ds, val_ds, test_ds, classes = get_datasets()
    kw = dict(batch_size=C.BATCH_SIZE, num_workers=C.NUM_WORKERS, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, drop_last=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
        classes,
    )