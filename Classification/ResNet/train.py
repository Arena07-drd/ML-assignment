import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import ResNet50

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "trashnet" / "dataset-resized"
RESULTS = HERE.parent / "results"  # shared with the DT/KNN/PCA experiments
SEED = 42
BATCH = 32
IMG = 224
WORKERS = 1

# ---------- preprocessing ----------
# ImageNet mean/std: the pretrained weights expect inputs normalized this way
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# train: random augmentation so the model sees a slightly different image every epoch
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    normalize,
])

# val/test: deterministic, no randomness
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG),
    transforms.ToTensor(),
    normalize,
])


def make_splits():
    """70% train / 10% val / 20% test, stratified and reproducible."""
    full = ImageFolder(DATA)  # only used for .targets and .classes
    idx = list(range(len(full)))
    trainval_idx, test_idx = train_test_split(
        idx, test_size=0.2, stratify=full.targets, random_state=SEED
    )
    # 12.5% of the 80% = 10% of the total
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=0.125,
        stratify=[full.targets[i] for i in trainval_idx], random_state=SEED,
    )
    # separate ImageFolders so only the training set gets augmentation
    train_ds = Subset(ImageFolder(DATA, transform=train_tf), train_idx)
    val_ds = Subset(ImageFolder(DATA, transform=eval_tf), val_idx)
    test_ds = Subset(ImageFolder(DATA, transform=eval_tf), test_idx)
    return full.classes, [full.targets[i] for i in train_idx], train_ds, val_ds, test_ds


def run_epoch(model, loader, criterion, device, opt=None):
    """One pass over loader: trains if opt is given, otherwise evaluates."""
    model.train(opt is not None)
    loss_sum, correct = 0.0, 0
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(x)
                loss = criterion(out, y)
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return loss_sum / n, correct / n


if __name__ == "__main__":  # required on Windows: DataLoader workers re-import this file
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    classes, train_targets, train_ds, val_ds, test_ds = make_splits()
    print("sizes -> train/val/test:", len(train_ds), len(val_ds), len(test_ds))
    train_loader = DataLoader(train_ds, BATCH, shuffle=True, num_workers=WORKERS)
    val_loader = DataLoader(val_ds, BATCH, num_workers=WORKERS)
    test_loader = DataLoader(test_ds, BATCH, num_workers=WORKERS)

    # inverse-frequency class weights so the small `trash` class isn't ignored
    counts = torch.bincount(torch.tensor(train_targets))
    weights = (len(train_targets) / (len(classes) * counts)).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    model = ResNet50(num_classes=len(classes), zero_init_residual=True).to(device)

    epochs = 100
    opt = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    RESULTS.mkdir(exist_ok=True)
    best, ckpt, hist, t0 = 0.0, HERE / "best_scratch.pt", [], time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, opt)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        sched.step()
        print(f"epoch {epoch:3d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f}", flush=True)
        hist.append([epoch, tr_loss, tr_acc, va_loss, va_acc])
        if va_acc > best:  # keep the best-val checkpoint
            best = va_acc
            torch.save(model.state_dict(), ckpt)
    print(f"training time: {(time.time() - t0) / 60:.1f} min")
    pd.DataFrame(hist, columns=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]).to_csv(
        RESULTS / "cnn_history.csv", index=False)

    # test set is only touched once, with the best-val checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            probs += torch.softmax(model(x.to(device)).float(), 1).cpu().tolist()
            labels += y.tolist()
    probs = np.array(probs)
    np.savez(RESULTS / "cnn_test.npz", y=np.array(labels), proba=probs)  # for ROC / comparison with DT, KNN
    preds = probs.argmax(1)
    print(classification_report(labels, preds, target_names=classes, digits=3))
    print(confusion_matrix(labels, preds))
