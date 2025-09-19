# asb – Deep Learning Training & Fine‑Tuning

A clean, reproducible PyTorch project for training, evaluating, fine‑tuning, and (optionally) quantizing image‑classification models on common benchmarks (e.g., CIFAR‑10/100, ImageNet). The repo includes reference training scripts, model definitions, and a quickstart workflow that runs out of the box.

> **Note:** Author names are intentionally omitted per project preference.

---

## ✨ Features
- **Plug‑and‑play training** via `main.py` (single‑file entrypoint).
- **Fine‑tuning** utilities via `fine_tune.py` (resume or transfer‑learn).
- **Model zoo scaffold** (e.g., MobileNetV2, ResNet‑style, ViT placeholder).
- **Configurable datasets**: CIFAR‑10/100, ImageNet (custom datasets easy to add).
- **Evaluation**: top‑1/top‑5 accuracy, confusion matrix, checkpoints.
- **(Optional) Quantization** notes in `QUANTIZATION_GUIDE.md`.
- **Lightweight dependencies** listed in `requirements.txt`.

---

## 🗂️ Repository Structure
```
asb/
├─ net_models/           # CNN & ViT model definitions (scaffold)
├─ ViT-pytorch/          # Vision Transformer-related helpers (if used)
├─ main.py               # Train/eval entrypoint
├─ fine_tune.py          # Fine‑tune/transfer‑learn entrypoint
├─ requirements.txt      # Python deps
├─ QUANTIZATION_GUIDE.md # Notes/instructions for post‑training quantization
├─ scheme.png            # (Optional) diagram/architecture sketch
└─ README.md
```
> Folder names may evolve; check the tree for the most up‑to‑date layout.

---

## ⚙️ Requirements
- Python 3.10+ (3.9 may work)
- PyTorch & torchvision compatible with your CUDA / CPU setup
- See `requirements.txt` for the exact packages and versions

Install dependencies:
```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1) Train from scratch
Example on **CIFAR‑100** with **MobileNetV2**:
```bash
python3 ./main.py --model MobileNetV2 --dataset cifar100
```

Common flags (typical examples):
```bash
# data & I/O
--data_root /path/to/data        # root folder that contains cifar/imagenet
--work_dir  ./runs/exp1          # where to store logs & checkpoints
--seed 42                         # reproducibility

# optimization
--epochs 200
--batch_size 128
--lr 0.1                          # base lr for from‑scratch training
--momentum 0.9
--weight_decay 1e-4
--sched step --milestones 100 150 --gamma 0.1   # or cosine, etc.

# model
--model MobileNetV2               # e.g., MobileNetV2, ResNet18, ViT
--pretrained false                # use pretrained weights if available

# evaluation
--eval_only false                 # set true to only run evaluation
--resume ./runs/exp1/ckpt.pt      # resume from checkpoint
```

### 2) Fine‑tune an existing checkpoint
```bash
python3 ./fine_tune.py   --model MobileNetV2   --dataset cifar100   --checkpoint /path/to/ckpt.pt   --lr 0.01 --epochs 50 --batch_size 256
```

### 3) Evaluate a checkpoint
```bash
python3 ./main.py   --model MobileNetV2   --dataset cifar100   --eval_only true   --resume /path/to/best.pt
```

---

## 📊 Datasets
- **CIFAR‑10 / CIFAR‑100**: will be auto‑downloaded by torchvision if not present (set `--data_root`).
- **ImageNet‑1k**: provide the prepared folder structure (`train/`, `val/`).

> To add a **custom dataset**, plug your `torch.utils.data.Dataset` into the data loader code where CIFAR/ImageNet are selected.

---

## 🧠 Models
The repo provides a minimal, extensible scaffold. Typical options include:
- **MobileNetV2** (fast, mobile‑friendly)
- **ResNet‑18/34** (strong baselines)
- **ViT** (Vision Transformer; experimental helpers under `ViT-pytorch/`)

You can add your own models under `net_models/` and expose them via a factory in the entry script.

---

## 🏋️ Training Tips (Recommended Defaults)
- **Optimizer**: SGD with momentum 0.9, weight decay `1e-4` for vision tasks.
- **LR schedule**: Step (drops ×0.1 at 100/150 epochs) or cosine decay.
- **Epochs**: 200 for CIFAR‑10/100; adjust for ImageNet according to resources.
- **Batch size**: 128 baseline; 256 if memory permits (especially for fine‑tuning).
- **Mixed precision**: Enable `torch.cuda.amp` if supported to speed up training.

---

## 🧪 Reproducibility
- Set `--seed` (e.g., 42) and `torch.backends.cudnn.deterministic = True` where appropriate.
- Log **exact** `requirements.txt` versions and the Git commit hash for each run.
- Save checkpoints and final metrics under a unique `--work_dir` per run.

---

## 🧱 Quantization (Optional)
See `QUANTIZATION_GUIDE.md` for notes on post‑training quantization and calibration. Start from a well‑trained FP32 model and measure accuracy drop vs. size/latency gains.

---

## 📦 Exporting / Inference
Typical export steps (adapt as needed):
```python
# example: load a trained checkpoint and run inference
import torch
from net_models import create_model  # example factory

model = create_model("MobileNetV2", num_classes=100)
ckpt = torch.load("runs/exp1/best.pt", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()

# single image inference (CIFAR‑style 32×32)
x = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    logits = model(x)
    probs = logits.softmax(dim=1)
print(probs.topk(5))
```

---

## 🧰 Troubleshooting
- **Low accuracy**: Verify preprocessing, LR schedule, and that `num_classes` matches the dataset.
- **CUDA OOM**: Reduce `--batch_size`, use gradient accumulation, or enable AMP.
- **Dataloader stalls**: Set `num_workers` per your CPU, try `pin_memory=True`.
- **Checkpoint mismatch**: Ensure the model name/arch matches the checkpoint metadata.

---

## 🔧 Development Notes
- Keep PRs small and focused.
- Add unit tests for new models/losses.
- Run `flake8`/`black` (or preferred linter/formatter) before committing.

---

## 📜 License
This project is licensed under the **Apache‑2.0** License. See `LICENSE` for details.

---

## 🙏 Acknowledgments
This project builds on the awesome PyTorch ecosystem and community examples. (Attribution intentionally anonymized.)
