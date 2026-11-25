"""Central configuration for the Oxford-IIIT Pet classification project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # project root (src/ is one level down)
DATA_DIR = ROOT / "data"
REAL_DIR = ROOT / "real_photos"

OUT_DIR = ROOT / "outputs"
CKPT_DIR = OUT_DIR / "checkpoints"
FIG_DIR = OUT_DIR / "figures"
METRIC_DIR = OUT_DIR / "metrics"
CKPT_PATH = CKPT_DIR / "best_model.pt"

for _d in (CKPT_DIR, FIG_DIR, METRIC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Data ----
IMG_SIZE = 224          # ResNet-50 ImageNet input size
VAL_RATIO = 0.15        # part of the official "trainval" split used for validation
NUM_CLASSES = 37
SEED = 42
NUM_WORKERS = 4         # set to 0 if DataLoader hangs on Windows

# ImageNet normalization (pretrained backbone expects this)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# ---- Training ----
BATCH_SIZE = 32         # fits in 6 GB VRAM with AMP
EPOCHS = 15
LR_HEAD = 1e-3          # new classifier layer learns faster
LR_BACKBONE = 1e-4      # pretrained layers are only fine-tuned
WEIGHT_DECAY = 1e-2     # AdamW decoupled weight decay
LABEL_SMOOTHING = 0.1
PATIENCE = 5            # early stopping on val accuracy
USE_AMP = True          # mixed precision on GPU