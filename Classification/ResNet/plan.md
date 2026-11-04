# ResNet-50 on TrashNet — transfer learning vs from scratch

## Context
Classify TrashNet images (6 classes) with ResNet-50 two ways and compare: **(A) transfer learning** (ImageNet-pretrained weights) and **(B) training from scratch** (random init). User is new to CV, so one heavily-commented script with labelled steps; the two modes differ only in model init + training schedule.

Findings from exploring the repo:
- `Classification/trashnet/dataset-resized/<class>/*.jpg` is already in `ImageFolder` layout; images are 512x384.
- 2527 images: cardboard 403, glass 501, metal 410, paper 594, plastic 482, **trash 137** (imbalanced).
- `Classification/ResNet/` is empty. `pyproject.toml` has only datasets, pillow, scikit-learn (no torch).
- User has an **RTX 4050 (6 GB)**; `nvidia-smi` failed with a driver/permission error that a reboot should clear. Plan targets CUDA with mixed precision; script falls back to CPU if CUDA is unavailable.
- `pyproject.toml` bug: `requires`/`build-backend` sit under `[project.scripts]` with no `[build-system]` header, and `ml-project = "ml_project:main"` points at a nonexistent package. Fix (add `[build-system]` header, drop the script entry) only if `uv add` complains.

## Steps

### 0. Dependencies (CUDA build of PyTorch)
Default PyPI wheels on Windows are CPU-only, so point uv at the PyTorch CUDA index in `pyproject.toml`:
```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu126" }
torchvision = { index = "pytorch-cu126" }
```
then `uv add torch torchvision matplotlib`. Add `*.pt` to `.gitignore`. After the reboot confirm `torch.cuda.is_available()` is `True` (first thing to check).

### 1. Single file: `Classification/ResNet/train.py`
CLI: `uv run Classification/ResNet/train.py --mode transfer|scratch`. Everything in `if __name__ == "__main__":` (required for DataLoader workers on Windows).

**a. Config:** `DATA` resolved relative to the file, `SEED=42`, `BATCH=32`, `IMG=224`, `WORKERS=4`; `torch.manual_seed` + `random_state=SEED` for reproducibility.

**b. Split (70/15/15, stratified):** load `ImageFolder(DATA)` only for `.targets`; `train_test_split` twice on indices with `stratify=targets`. Split by index so val/test never get augmentation. Same split for both modes → fair comparison.

**c. Preprocessing:**
- Train: `RandomResizedCrop(224, scale=(0.6,1.0))` → `RandomHorizontalFlip` → `RandomVerticalFlip` (trash has no fixed "up") → `RandomRotation(15)` → `ColorJitter(0.2,0.2,0.2)` → `ToTensor` → `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`.
- Val/Test: `Resize(256)` → `CenterCrop(224)` → `ToTensor` → same `Normalize`.
- Two `ImageFolder` objects (train / eval transform) wrapped in `Subset(ds, idx)`.
- ImageNet stats are used in both modes: required for transfer, harmless for scratch.

**d. Loaders:** batch 32, `shuffle=True` only for train, `num_workers=4`, `pin_memory=True`, `persistent_workers=True`.

**e. Loss (both modes):** `CrossEntropyLoss(weight=inverse-frequency class weights from train labels, label_smoothing=0.1)`.

**f. Speed on the 4050 (both modes):** `torch.autocast("cuda", dtype=torch.float16)` + `GradScaler`, `model.to(memory_format=torch.channels_last)`. Batch 32 at 224px fits in 6 GB with AMP.

**g. Mode A — transfer:** `resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`; `fc = Sequential(Dropout(0.2), Linear(2048, 6))`.
1. *Head only*, 5 epochs: freeze everything but `fc` (backbone kept in `.eval()` so BatchNorm stats don't drift); `AdamW(lr=1e-3, weight_decay=1e-4)`, constant LR.
2. *Fine-tune all*, up to 15 epochs (early stop if val acc doesn't improve for 5): unfreeze everything; AdamW param groups backbone `lr=1e-4`, head `lr=1e-3`, `weight_decay=1e-4`, `CosineAnnealingLR` to ~0. (SGD alternative: momentum 0.9, backbone lr 1e-3, head lr 1e-2, wd 5e-4.)

**h. Mode B — scratch:** `resnet50(weights=None, zero_init_residual=True)` (He/Kaiming init as in the paper; `zero_init_residual` is from Goyal et al. 2017), same new 6-way `fc` with Dropout 0.2, all layers trainable from epoch 1. `SGD(lr=0.02, momentum=0.9, nesterov=True, weight_decay=5e-4)`, 5-epoch linear warmup (0.002 → 0.02) then cosine to 0, **100 epochs** (random init needs far more passes). LR range worth trying: 0.01–0.05 (the paper's 0.1 at batch 256 scales to 0.0125 at batch 32). Regularisation: augmentation + label smoothing + weight decay + dropout; add mixup/cutmix only if train acc hits ~100% while val stalls.

**Tuning guide after first run:** overfit (train >> val) → more weight decay / stronger aug / lower LR; underfit → LR x2–3 or train longer; NaN/loss spikes → LR /2–3; poor `trash` recall → raise its class weight.

**i. Loop per epoch:** train → val (`no_grad`, `eval()`) → print train/val loss+acc → save `best_{mode}.pt` on best val acc → append row to `history_{mode}.csv`.

**j. Final evaluation (test set, once per mode):** load best checkpoint; print `classification_report`; save `confusion_matrix_{mode}.png` and `curves_{mode}.png` (loss/acc vs epoch) in `Classification/ResNet/`.

### 2. Comparison
After both runs, print a small table: mode, best val acc, test acc, macro-F1, epochs, train time. Expected story: transfer converges in a few epochs to ~92–96%; scratch needs many epochs and lands noticeably lower (~60–80%) because 1.8k images can't teach 25M parameters from nothing — that gap is the point of the experiment.

### 3. Optional follow-ups (not in first pass)
Grad-CAM to visualise what each model looks at; a smaller scratch net (e.g. ResNet-18) as a fairer from-scratch baseline.

## Verification
1. `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` → `True, ...RTX 4050...`.
2. Startup sanity prints: split sizes (~1768/379/380), per-class counts per split, one batch shape `[32,3,224,224]`, normalized batch mean≈0 / std≈1.
3. Smoke test: `--mode transfer` with 1+1 epochs and a ~64-image subset runs end-to-end and writes checkpoint, CSV, PNGs; same for `--mode scratch` with 2 epochs.
4. Full runs: transfer (~20 epochs, a few minutes on GPU) and scratch (100 epochs, tens of minutes). Check train acc rises gradually and the confusion matrices make sense (`trash`/`glass`/`plastic` likely confusions).
