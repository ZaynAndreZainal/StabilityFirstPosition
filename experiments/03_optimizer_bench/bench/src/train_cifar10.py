from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import torchvision
    import torchvision.transforms as T
except Exception as e:
    raise RuntimeError("torchvision is required for CIFAR-10 scripts") from e

from log_results import ResultLogger, WallClock


# -----------------
# Config utilities
# -----------------

def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_merged_configs(paths: List[str]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for p in paths:
        if not p:
            continue
        deep_update(cfg, load_yaml(p))
    return cfg


# -----------------
# Model
# -----------------

def build_model(cfg: Dict[str, Any]) -> nn.Module:
    name = cfg["model"]["name"].lower()
    if name == "resnet18":
        model = torchvision.models.resnet18(num_classes=int(cfg["model"].get("num_classes", 10)))
        # CIFAR-10: replace first conv/maxpool for small images
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    raise ValueError(f"Unknown model: {name}")


# -----------------
# Optimizers
# -----------------

class SAM(optim.Optimizer):
    """Minimal SAM wrapper.

    Reference: Foret et al. (2021). This implementation expects a closure.
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        if rho <= 0.0:
            raise ValueError("rho must be positive")
        self.rho = rho
        self.adaptive = adaptive
        self.base_optimizer = base_optimizer(params, **kwargs)
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(self.base_optimizer.param_groups, defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p].get("e_w", 0.0))
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires closure"
        closure = torch.enable_grad()(closure)
        loss = closure()
        self.first_step(zero_grad=True)
        closure()
        self.second_step(zero_grad=True)
        return loss

    def _grad_norm(self):
        device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if self.adaptive:
                    norms.append((torch.abs(p) * p.grad).norm(p=2))
                else:
                    norms.append(p.grad.norm(p=2))
        if not norms:
            return torch.tensor(0.0, device=device)
        return torch.norm(torch.stack(norms), p=2)


def build_optimizer(cfg: Dict[str, Any], model: nn.Module):
    """Optimizer factory for CIFAR-10.

    Supported (minimal + runnable):
      - sgd
      - sgd_momentum
      - adam
      - adamw
      - rmsprop
      - lion
      - sam_sgd
      - sam_adamw

    Notes:
      - 'sam_sgd' uses an SGD-momentum base optimizer.
      - 'sam_adamw' uses an AdamW base optimizer.
    """

    opt_cfg = cfg["optimizer"]
    name = str(opt_cfg["name"]).lower()
    p = opt_cfg.get("params", {})

    if name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=float(p["lr"]),
            momentum=0.0,
            nesterov=False,
            weight_decay=float(p.get("weight_decay", 0.0)),
        )

    if name == "sgd_momentum":
        return optim.SGD(
            model.parameters(),
            lr=float(p["lr"]),
            momentum=float(p.get("momentum", 0.9)),
            nesterov=bool(p.get("nesterov", True)),
            weight_decay=float(p.get("weight_decay", 0.0)),
        )

    if name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=float(p["lr"]),
            betas=tuple(p.get("betas", [0.9, 0.999])),
            eps=float(p.get("eps", 1e-8)),
            weight_decay=float(p.get("weight_decay", 0.0)),
        )

    if name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=float(p["lr"]),
            betas=tuple(p.get("betas", [0.9, 0.999])),
            eps=float(p.get("eps", 1e-8)),
            weight_decay=float(p.get("weight_decay", 0.0)),
        )

    if name == "rmsprop":
        return optim.RMSprop(
            model.parameters(),
            lr=float(p["lr"]),
            alpha=float(p.get("alpha", 0.99)),
            eps=float(p.get("eps", 1e-8)),
            momentum=float(p.get("momentum", 0.0)),
            centered=bool(p.get("centered", False)),
            weight_decay=float(p.get("weight_decay", 0.0)),
        )

    if name == "lion":
        betas = tuple(p.get("betas", [0.9, 0.99]))
        lr = float(p["lr"])
        wd = float(p.get("weight_decay", 0.0))

        class Lion(optim.Optimizer):
            def __init__(self, params):
                super().__init__(params, dict(lr=lr, betas=betas, weight_decay=wd))

            @torch.no_grad()
            def step(self, closure=None):
                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()
                for group in self.param_groups:
                    lr_ = group["lr"]
                    b1, b2 = group["betas"]
                    wd_ = group["weight_decay"]
                    for p_ in group["params"]:
                        if p_.grad is None:
                            continue
                        g = p_.grad
                        if wd_ != 0:
                            p_.mul_(1 - lr_ * wd_)
                        state = self.state[p_]
                        if "m" not in state:
                            state["m"] = torch.zeros_like(p_)
                        m = state["m"]
                        m.mul_(b1).add_(g, alpha=1 - b1)
                        p_.add_(torch.sign(m), alpha=-lr_)
                        m.mul_(b2).add_(g, alpha=1 - b2)
                return loss

        return Lion(model.parameters())

    if name == "sam_sgd":
        rho = float(p.get("rho", 0.05))
        adaptive = bool(p.get("adaptive", False))
        base_p = p.get("base_params", {})

        def base_opt(params, **kwargs):
            return optim.SGD(params, **kwargs)

        return SAM(
            model.parameters(),
            base_optimizer=base_opt,
            rho=rho,
            adaptive=adaptive,
            lr=float(base_p.get("lr", p.get("lr", 0.1))),
            momentum=float(base_p.get("momentum", 0.9)),
            nesterov=bool(base_p.get("nesterov", True)),
            weight_decay=float(base_p.get("weight_decay", p.get("weight_decay", 0.0))),
        )

    if name == "sam_adamw":
        rho = float(p.get("rho", 0.05))
        adaptive = bool(p.get("adaptive", False))
        base_p = p.get("base_params", {})

        def base_opt(params, **kwargs):
            return optim.AdamW(params, **kwargs)

        return SAM(
            model.parameters(),
            base_optimizer=base_opt,
            rho=rho,
            adaptive=adaptive,
            lr=float(base_p.get("lr", p.get("lr", 1e-3))),
            betas=tuple(base_p.get("betas", p.get("betas", [0.9, 0.999]))),
            eps=float(base_p.get("eps", p.get("eps", 1e-8))),
            weight_decay=float(base_p.get("weight_decay", p.get("weight_decay", 0.0))),
        )

    raise ValueError(
        f"Unknown optimizer: {name}. Supported: sgd, sgd_momentum, adam, adamw, rmsprop, lion, sam_sgd, sam_adamw"
    )


def build_scheduler(cfg: Dict[str, Any], optimizer):
    sch = cfg.get("scheduler", {"name": "none"})
    name = sch.get("name", "none").lower()
    p = sch.get("params", {})

    if name in ("none", "null"):
        return None

    if name == "cosine":
        # Step per epoch
        min_lr = float(p.get("min_lr", 0.0))
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["training"]["max_epochs"]), eta_min=min_lr)

    raise ValueError(f"Unknown scheduler: {name}")


# -----------------
# Training / eval
# -----------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
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


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, *, epoch: int = 0):
    model.train()
    running = 0.0

    for x, y in tqdm(loader, desc=f"epoch {epoch}", leave=False, unit="batch"):
        x = x.to(device)
        y = y.to(device)

        def closure():
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                logits = model(x)
                loss = criterion(logits, y)
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
            return loss

        if optimizer.__class__.__name__.lower() == "sam":
            loss = optimizer.step(closure)
        else:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                logits = model(x)
                loss = criterion(logits, y)
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # Abort on non-finite loss (prevents NaN cascades that later show up as chance-accuracy tails)
        loss_val = float(loss.detach().cpu().item())
        if not np.isfinite(loss_val):
            # Signal failure to caller by returning NaN
            return float("nan")

        running += loss_val

    return running / max(len(loader), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=[], help="YAML config(s) to merge")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    cfg = load_merged_configs(args.config)
    cfg.setdefault("training", {})
    cfg["training"]["seed"] = args.seed

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    root = cfg["data"].get("root", "data/cifar10")
    train_ds = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    test_ds = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True,
                              num_workers=int(cfg["data"].get("num_workers", 8)), pin_memory=bool(cfg["data"].get("pin_memory", True)))
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

    model = build_model(cfg).to(device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)))

    use_amp = bool(cfg.get("training", {}).get("mixed_precision", True)) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    logger = ResultLogger(args.out_dir, filename=cfg.get("logging", {}).get("json_filename", "metrics.json"))
    clock = WallClock()

    best_acc = -1.0
    best_epoch = -1
    time_to_95 = None

    # Trajectory logging for accuracy-over-time plots
    # Stored into metrics.json so you can aggregate/plot learning curves later.
    curve_t_s = []
    curve_acc = []
    curve_train_loss = []

    # Fixed time-to-target (for Pareto efficiency plots)
    # Aggregated downstream as summary['time_to_target_s_mean'].
    target_acc = float(cfg.get("training", {}).get("target_acc", 0.90))
    stop_on_target = bool(cfg.get("training", {}).get("stop_on_target", False))
    time_to_target = None

    max_epochs = int(cfg["training"].get("max_epochs", 200))
    wallclock_seconds = float(cfg.get("budget", {}).get("wallclock_seconds", float("inf")))

    epochs_pbar = tqdm(range(max_epochs), desc="training", unit="epoch")
    for epoch in epochs_pbar:
        if clock.seconds() > wallclock_seconds:
            break

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler, epoch=epoch)
        acc = evaluate(model, test_loader, device)

        # Abort run on non-finite metrics (prevents misleading curve tails)
        if not np.isfinite(float(train_loss)) or not np.isfinite(float(acc)):
            break

        # Append learning curve point
        curve_t_s.append(float(clock.seconds()))
        curve_acc.append(float(acc))
        curve_train_loss.append(float(train_loss))
        if scheduler is not None:
            scheduler.step()

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

        # Fixed time-to-target (first time reaching a pre-specified accuracy)
        if time_to_target is None and acc >= target_acc:
            time_to_target = clock.seconds()
            if stop_on_target:
                break

        # time-to-threshold defined relative to final best in-run; approximate online by using current best
        if best_acc > 0:
            thr = 0.95 * best_acc
            if time_to_95 is None and acc >= thr:
                time_to_95 = clock.seconds()

        epochs_pbar.set_postfix(
            loss=f"{train_loss:.4f}",
            acc=f"{acc:.4f}",
            best=f"{best_acc:.4f}",
            t_s=f"{clock.seconds():.1f}",
        )

    payload = {
        "domain": "cifar10",
        "task": "classification",
        "regime": cfg.get("budget", {}).get("regime", "fixed_time"),
        "budget": cfg.get("budget", {}),
        "optimizer": cfg.get("optimizer", {}).get("name"),
        "seed": int(args.seed),
        "iid_score": float(best_acc),
        "time_to_threshold_s": None if time_to_95 is None else float(time_to_95),
        "time_to_target_s": {"acc@0.90": None if time_to_target is None else float(time_to_target)},
        "curve": {
            "t_s": curve_t_s,
            "acc": curve_acc,
            "train_loss": curve_train_loss,
        },
        "sharpness": None,  # optional: fill via a post-hoc estimator
        "notes": {
            "best_epoch": int(best_epoch),
            "wallclock_seconds": float(clock.seconds()),
        },
    }
    logger.save(payload)


if __name__ == "__main__":
    main()