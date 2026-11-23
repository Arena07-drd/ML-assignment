"""Evaluate the saved best_scratch.pt on the held-out test split (same split as train.py)."""
import sys

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from model import ResNet50
from train import BATCH, HERE, RESULTS, WORKERS, make_splits

if __name__ == "__main__":  # required on Windows: DataLoader workers re-import this file
    ckpt = sys.argv[1] if len(sys.argv) > 1 else HERE / "best_scratch.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes, _, _, _, test_ds = make_splits()
    loader = DataLoader(test_ds, BATCH, num_workers=WORKERS)

    model = ResNet50(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            probs += torch.softmax(model(x.to(device)).float(), 1).cpu().tolist()
            labels += y.tolist()
    probs = np.array(probs)
    preds = probs.argmax(1)

    print(f"test samples: {len(labels)} | accuracy: {(preds == np.array(labels)).mean():.4f}")
    print(classification_report(labels, preds, target_names=classes, digits=3))
    print(confusion_matrix(labels, preds))
    np.savez(RESULTS / "cnn_test.npz", y=np.array(labels), proba=probs)
