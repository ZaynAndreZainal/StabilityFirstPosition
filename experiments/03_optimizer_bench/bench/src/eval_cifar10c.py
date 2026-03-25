from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import torchvision
    import torchvision.transforms as T
except Exception as e:
    raise RuntimeError("torchvision is required") from e


def load_cifar10c(root: str, corruption: str, severity: int):
    # CIFAR-10-C standard format: root/{corruption}.npy and root/labels.npy
    x = np.load(os.path.join(root, f"{corruption}.npy"))
    y = np.load(os.path.join(root, "labels.npy"))

    # Each corruption file has 5 severities concatenated, each of length 10k.
    n = 10000
    idx0 = (severity - 1) * n
    idx1 = severity * n
    x = x[idx0:idx1]
    y = y[idx0:idx1]
    return x, y


@torch.no_grad()
def eval_accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.numel())
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to a torch saved model state_dict")
    ap.add_argument("--cifar10c_root", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--corruptions", type=str, default="all")
    ap.add_argument("--severities", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.to(device)

    # Corruptions list
    all_corr = [
        "gaussian_noise", "shot_noise", "impulse_noise",
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
        "snow", "frost", "fog", "brightness",
        "contrast", "elastic_transform", "pixelate", "jpeg_compression",
        "speckle_noise", "gaussian_blur", "spatter", "saturate",
    ]
    corruptions = all_corr if args.corruptions == "all" else [c.strip() for c in args.corruptions.split(",")]

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    results: Dict[str, Any] = {"per_corruption": {}}
    accs = []

    for corr in corruptions:
        per_sev = []
        for sev in args.severities:
            x, y = load_cifar10c(args.cifar10c_root, corr, sev)
            # x is uint8 NHWC
            x_t = torch.stack([transform(xi) for xi in x])
            y_t = torch.from_numpy(y).long()
            loader = DataLoader(TensorDataset(x_t, y_t), batch_size=256, shuffle=False)
            acc = eval_accuracy(model, loader, device)
            per_sev.append(acc)
        results["per_corruption"][corr] = {"per_severity": per_sev}
        accs.extend(per_sev)

    results["mean_acc"] = float(np.mean(accs)) if accs else None
    results["worst_acc"] = float(np.min(accs)) if accs else None

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()