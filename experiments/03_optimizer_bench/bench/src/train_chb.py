from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from log_results import ResultLogger, WallClock
from metrics import auc_roc, bce_logits_to_scores

import glob
import os
import numpy as np


# -----------------
# Config utilities
# -----------------


def _load_npz_shards_as_windows(split_dir: str) -> List[Any]:
  shards = sorted(glob.glob(os.path.join(split_dir, "shard_*.npz")))
  if not shards:
    raise FileNotFoundError(f"No shards found in {split_dir}")

  windows = []
  for p in shards:
    d = np.load(p)
    X = d["X"]
    y = d["y"]
    windows.extend([(X[i], int(y[i])) for i in range(X.shape[0])])
  return windows


def load_chb_splits(cfg: Dict[str, Any]) -> Tuple[List[Any], List[Any], List[Any]]:
  cache = cfg.get("chb", {}).get("cache", {})
  cache_root = cache.get("root", "/home/zzai2932/Data/CHB-MIT/cache")
  cache_name = cache.get("name", "chb_detection_v1")
  base = os.path.join(cache_root, cache_name)

  train = _load_npz_shards_as_windows(os.path.join(base, "train"))
  val = _load_npz_shards_as_windows(os.path.join(base, "val"))
  test = _load_npz_shards_as_windows(os.path.join(base, "test"))
  return train, val, test

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


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------
# CHB adapters (YOU WIRE THESE)
# -----------------

class ChbShardDataset(Dataset):
  """
  Streams cached CHB windows from shard_*.npz files without loading everything into RAM.

  Each shard contains:
    X: [N, C, T] float32
    y: [N] int64
  """

  def __init__(self, split_dir: str):
    self.split_dir = split_dir
    self.shards = sorted(glob.glob(os.path.join(split_dir, "shard_*.npz")))
    if not self.shards:
      raise FileNotFoundError(f"No shards found in {split_dir}")

    # Build an index: global_idx -> (shard_idx, row_idx)
    self._index = []
    self._shard_sizes = []
    for si, p in enumerate(self.shards):
      with np.load(p, allow_pickle=False) as d:
        n = int(d["X"].shape[0])
      self._shard_sizes.append(n)
      self._index.extend([(si, i) for i in range(n)])

    # small cache to avoid reloading the same shard repeatedly
    self._cache_shard_idx = None
    self._cache_X = None
    self._cache_y = None

  def __len__(self):
    return len(self._index)

  def _load_shard(self, shard_idx: int):
    if self._cache_shard_idx == shard_idx:
      return
    p = self.shards[shard_idx]
    d = np.load(p, allow_pickle=False)
    self._cache_shard_idx = shard_idx
    self._cache_X = d["X"]  # [N, C, T]
    self._cache_y = d["y"]  # [N]

  def __getitem__(self, idx: int):
    shard_idx, row_idx = self._index[idx]
    self._load_shard(shard_idx)
    x = self._cache_X[row_idx]
    y = self._cache_y[row_idx]
    # Keep return format consistent with your original dataset:
    # x: np.ndarray [C, T], y: scalar
    return x, float(y)


import hashlib
import os
import re
from typing import Dict, List, Tuple

_RE_FILE = re.compile(r"^File Name:\\s*(?P<name>\\S+\\.edf)\\s*$")
_RE_NSZ  = re.compile(r"^Number of Seizures in File:\\s*(?P<n>\\d+)\\s*$")
_RE_SSZ  = re.compile(r"^Seizure Start Time:\\s*(?P<t>\\d+)\\s*seconds\\s*$")
_RE_ESZ  = re.compile(r"^Seizure End Time:\\s*(?P<t>\\d+)\\s*seconds\\s*$")

# def parse_chb_summary(summary_path: str) -> Dict[str, List[Tuple[int, int]]]:
#     """
#     Returns mapping: edf_filename -> list of (start_sec, end_sec) seizure intervals.
#     """
#     with open(summary_path, "r", errors="ignore") as f:
#         lines = [ln.strip() for ln in f.readlines()]
#
#     out: Dict[str, List[Tuple[int, int]]] = {}
#     cur_file = None
#     expected = None
#     pending_starts: List[int] = []
#
#     for ln in lines:
#         m = _RE_FILE.match(ln)
#         if m:
#             cur_file = m.group("name")
#             out.setdefault(cur_file, [])
#             expected = None
#             pending_starts = []
#             continue
#
#         if cur_file is None:
#             continue
#
#         m = _RE_NSZ.match(ln)
#         if m:
#             expected = int(m.group("n"))
#             continue
#
#         m = _RE_SSZ.match(ln)
#         if m:
#             pending_starts.append(int(m.group("t")))
#             continue
#
#         m = _RE_ESZ.match(ln)
#         if m:
#             end_t = int(m.group("t"))
#             if not pending_starts:
#                 raise ValueError(f"Found end time without start time in {summary_path} for {cur_file}")
#             start_t = pending_starts.pop(0)
#             if end_t < start_t:
#                 raise ValueError(f"End < start in {summary_path} for {cur_file}: {start_t}..{end_t}")
#             out[cur_file].append((start_t, end_t))
#             continue
#
#     # Optional consistency check: if expected is present, verify count
#     # (Some summaries can be messy; you can downgrade this to warnings.)
#     for edf, intervals in out.items():
#         pass
#
#     return out

import glob
import hashlib
import os
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

# -------------------------
# Deterministic patient split
# -------------------------

def split_patient(patient_id: str, test_pct: int = 20, val_pct: int = 10) -> str:
    """
    Stable (non-salted) split based on MD5(patient_id).
    Returns: "train" | "val" | "test"
    """
    h = hashlib.md5(patient_id.encode("utf-8")).hexdigest()
    r = int(h[:8], 16) % 100
    if r < test_pct:
        return "test"
    if r < test_pct + val_pct:
        return "val"
    return "train"

def overlaps_any_interval(t0: float, t1: float, intervals: List[Tuple[int, int]]) -> int:

  """Intervals are (start_sec, end_sec). Window is [t0, t1) in seconds."""

  for s, e in intervals:
    if t0 < e and t1 > s:
      return 1
  return 0

# -------------------------
# EDF -> windows adapter surface
# -------------------------

# def edf_to_windows(edf_path: str,
#                    seizure_intervals_sec: List[Tuple[int, int]],
#                    *,
#                    window_seconds: float,
#                    stride_seconds: float,
#                    resample_hz: int,
#                    bandpass_hz: Tuple[float, float],
#                    notch_hz: Optional[float],
#                    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
#
#   """
#   Returns:
#   X: float32 [C, T_total] after preprocessing (resample/filter/notch)
#   y_win: int64 [N_windows]
#   win_starts_sec: float list length N_windows (or return as array if you prefer)  ch_names: list[str] length C
#   """
#   raise NotImplementedError("Implement EDF reading + preprocessing + windowing here.")

# -------------------------

# Canonical channel projection + mask

# -------------------------

def normalize_ch_name(s: str) -> str:
  return s.strip().upper().replace(" ", "")

# def build_canonical_channels(patient_dirs: List[str],
#                              *,
#                              min_freq: float,
#                              ) -> List[str]:
#   raise NotImplementedError("Implement EDF header scan to compute canonical channels.")

def project_to_canonical_with_mask(X: np.ndarray,
                                   ch_names: List[str],
                                   canonical: List[str],
                                   ) -> Tuple[np.ndarray, np.ndarray]:
  name_to_idx = {normalize_ch_name(n): i for i, n in enumerate(ch_names)}
  Cc = len(canonical)
  T = X.shape[1]
  Xc = np.zeros((Cc, T), dtype=np.float32)
  mask = np.zeros((Cc,), dtype=np.float32)
  for j, cname in enumerate(canonical):
    idx = name_to_idx.get(normalize_ch_name(cname))
    if idx is not None:
      Xc[j] = X[idx].astype(np.float32)
      mask[j] = 1.0
  return Xc, mask

def concat_mask_as_channels(Xc: np.ndarray, mask: np.ndarray) -> np.ndarray:
  """
  Convert mask [C] into [C, T] and concatenate => [2C, T].
  This lets your CNN remain simple (fixed input channels).
  """
  C, T = Xc.shape
  M = np.repeat(mask[:, None], T, axis=1).astype(np.float32)

  return np.concatenate([Xc, M], axis=0)

def load_chb_datasets(cfg: dict):

  cache = cfg.get("chb", {}).get("cache", {})
  cache_root = cache.get("root", "/home/zzai2932/Data/CHB-MIT/cache")
  cache_name = cache.get("name", "chb_detection_v1")

  base = os.path.join(cache_root, cache_name)

  train_ds = ChbShardDataset(os.path.join(base, "train"))
  val_ds   = ChbShardDataset(os.path.join(base, "val"))
  test_ds  = ChbShardDataset(os.path.join(base, "test"))

  return train_ds, val_ds, test_ds

# -----------------
# Model
# -----------------

class EegCnn1D(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        # x: [B, C, T]
        z = self.net(x).squeeze(-1)
        return self.head(z).squeeze(-1)  # [B]


# -----------------
# Optimizers
# -----------------

class SAM(optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
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
        assert closure is not None
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
    """Optimizer factory for CHB.

    Supported (minimal + runnable):
      - sgd
      - sgd_momentum
      - adam
      - adamw
      - rmsprop
      - lion
      - sam (CHB-style: params.base.name)
      - sam_sgd (CIFAR-style alias)
      - sam_adamw (CIFAR-style alias)

    Notes:
      - CHB-style SAM config uses: optimizer.name: sam, optimizer.params.base.{name,params}
      - CIFAR-style SAM aliases use: optimizer.name: sam_sgd|sam_adamw with optimizer.params.base_params
    """

    opt_cfg = cfg["optimizer"]
    name = str(opt_cfg["name"]).lower()
    p = opt_cfg.get("params", {}) or {}

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

    # -----------------
    # SAM (two schemas)
    # -----------------

    if name == "sam":
        rho = float(p.get("rho", 0.05))
        adaptive = bool(p.get("adaptive", False))
        base = p["base"]
        base_name = str(base["name"]).lower()
        base_p = base.get("params", {}) or {}

        if base_name == "sgd_momentum":

            def base_opt(params, **kwargs):
                return optim.SGD(params, **kwargs)

            return SAM(
                model.parameters(),
                base_optimizer=base_opt,
                rho=rho,
                adaptive=adaptive,
                lr=float(base_p["lr"]),
                momentum=float(base_p.get("momentum", 0.9)),
                nesterov=bool(base_p.get("nesterov", True)),
                weight_decay=float(base_p.get("weight_decay", 0.0)),
            )

        if base_name == "adamw":

            def base_opt(params, **kwargs):
                return optim.AdamW(params, **kwargs)

            return SAM(
                model.parameters(),
                base_optimizer=base_opt,
                rho=rho,
                adaptive=adaptive,
                lr=float(base_p["lr"]),
                betas=tuple(base_p.get("betas", [0.9, 0.999])),
                eps=float(base_p.get("eps", 1e-8)),
                weight_decay=float(base_p.get("weight_decay", 0.0)),
            )

        raise ValueError(f"Unsupported SAM base optimizer: {base_name}")

    # CIFAR-style alias: sam_sgd
    if name == "sam_sgd":
        rho = float(p.get("rho", 0.05))
        adaptive = bool(p.get("adaptive", False))
        base_p = p.get("base_params", {}) or {}

        def base_opt(params, **kwargs):
            return optim.SGD(params, **kwargs)

        return SAM(
            model.parameters(),
            base_optimizer=base_opt,
            rho=rho,
            adaptive=adaptive,
            lr=float(base_p["lr"]),
            momentum=float(base_p.get("momentum", 0.9)),
            nesterov=bool(base_p.get("nesterov", True)),
            weight_decay=float(base_p.get("weight_decay", 0.0)),
        )

    # CIFAR-style alias: sam_adamw
    if name == "sam_adamw":
        rho = float(p.get("rho", 0.05))
        adaptive = bool(p.get("adaptive", False))
        base_p = p.get("base_params", {}) or {}

        def base_opt(params, **kwargs):
            return optim.AdamW(params, **kwargs)

        return SAM(
            model.parameters(),
            base_optimizer=base_opt,
            rho=rho,
            adaptive=adaptive,
            lr=float(base_p["lr"]),
            betas=tuple(base_p.get("betas", [0.9, 0.999])),
            eps=float(base_p.get("eps", 1e-8)),
            weight_decay=float(base_p.get("weight_decay", 0.0)),
        )

    raise ValueError(f"Unknown optimizer: {name}")


# -----------------
# Train / eval
# -----------------

@torch.no_grad()
def eval_auc(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    ys = []
    ps = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        p = torch.sigmoid(logits)
        ys.append(y.detach().cpu().numpy())
        ps.append(p.detach().cpu().numpy())
    y_true = np.concatenate(ys)
    y_score = np.concatenate(ps)
    return float(auc_roc(y_true, y_score))


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, grad_clip_norm: float = 0.0, *, epoch: int = 0):
    model.train()
    running = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc=f"epoch {epoch}", leave=False, unit="batch"):
        x = x.to(device)
        y = y.to(device)
        # ensure BCE targets are float for BCEWithLogitsLoss
        y = y.float()

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

        logits = None  # for train-acc logging

        if optimizer.__class__.__name__.lower() == "sam":
            # SAM path: optimizer.step runs forward/backward inside the closure.
            # We do a cheap post-step forward pass (no_grad) to log train accuracy.
            loss = optimizer.step(closure)
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=scaler is not None):
                    logits = model(x)
        else:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                logits = model(x)
                loss = criterion(logits, y)
            if scaler is None:
                loss.backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

        # Abort on non-finite loss (prevents NaN cascades that later show up as chance-metric tails)
        loss_val = float(loss.detach().cpu().item())
        if not np.isfinite(loss_val):
            return float("nan"), float("nan")

        running += loss_val

        # Training accuracy (binary classification; threshold 0.5)
        if logits is not None:
            with torch.no_grad():
                p = torch.sigmoid(logits.detach())
                preds = (p >= 0.5).to(y.dtype)
                correct += int((preds == y).sum().item())
                total += int(y.numel())

    train_loss = running / max(len(loader), 1)
    train_acc = correct / max(total, 1)
    return train_loss, train_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    cfg = load_merged_configs(args.config)
    cfg.setdefault("training", {})
    cfg["training"]["seed"] = args.seed

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds, val_ds, test_ds = load_chb_datasets(cfg)

    # infer channels
    x0, _ = train_ds[0]
    in_ch = int(x0.shape[0])

    model = EegCnn1D(in_channels=in_ch).to(device)
    optimizer = build_optimizer(cfg, model)

    use_amp = bool(cfg.get("training", {}).get("mixed_precision", True)) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    batch_size = int(cfg["training"].get("batch_size", 256))
    # NOTE: shuffle=False is recommended for shard-backed datasets to avoid shard thrashing / worker OOM.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=int(cfg["data"].get("num_workers", 8)), pin_memory=bool(cfg["data"].get("pin_memory", True)))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # loss
    # If you use imbalance weighting, set pos_weight in cfg and implement here.
    criterion = nn.BCEWithLogitsLoss()

    logger = ResultLogger(args.out_dir, filename=cfg.get("logging", {}).get("json_filename", "metrics.json"))
    clock = WallClock()

    best_val = -1.0
    best_test = -1.0
    best_epoch = -1

    # Trajectory logging for learning-curve plots
    curve_t_s = []
    curve_train_acc = []
    curve_val_auc = []
    curve_test_auc = []
    curve_train_loss = []

    # Fixed time-to-target (for Pareto efficiency plots)
    target_auc = float(cfg.get("training", {}).get("target_auc", 0.90))
    stop_on_target = bool(cfg.get("training", {}).get("stop_on_target", False))
    time_to_target = None

    patience = int(cfg["training"].get("early_stopping_patience_epochs", 10))
    do_es = bool(cfg["training"].get("early_stopping", True))
    es_count = 0

    # budget
    wallclock_seconds = float(cfg.get("budget", {}).get("wallclock_seconds", float("inf")))
    max_epochs = int(cfg["training"].get("max_epochs", 50))

    # time-to-threshold: define threshold against best_val achieved online
    time_to_95 = None

    epochs_pbar = tqdm(range(max_epochs), desc="training", unit="epoch")
    for epoch in epochs_pbar:
        if clock.seconds() > wallclock_seconds:
            break

        tr_loss, tr_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            grad_clip_norm=float(cfg["training"].get("grad_clip_norm", 1.0)),
            epoch=epoch,
        )
        val_auc = eval_auc(model, val_loader, device)
        test_auc = eval_auc(model, test_loader, device)

        # Abort run on non-finite metrics (prevents misleading curve tails)
        if not np.isfinite(float(tr_loss)) or not np.isfinite(float(tr_acc)) or not np.isfinite(float(val_auc)) or not np.isfinite(float(test_auc)):
            break

        # Append learning curve point
        curve_t_s.append(float(clock.seconds()))
        curve_train_acc.append(float(tr_acc))
        curve_val_auc.append(float(val_auc))
        curve_test_auc.append(float(test_auc))
        curve_train_loss.append(float(tr_loss))

        if val_auc > best_val:
            best_val = val_auc
            best_test = test_auc
            best_epoch = epoch
            es_count = 0
        else:
            es_count += 1

        # Fixed time-to-target: use validation AUC as the gating metric
        if time_to_target is None and val_auc >= target_auc:
            time_to_target = clock.seconds()
            if stop_on_target:
                break

        thr = 0.95 * best_val if best_val > 0 else None
        if thr is not None and time_to_95 is None and val_auc >= thr:
            time_to_95 = clock.seconds()

        epochs_pbar.set_postfix(
            loss=f"{tr_loss:.4f}",
            val_auc=f"{val_auc:.4f}",
            best_val=f"{best_val:.4f}",
            test_auc=f"{test_auc:.4f}",
            t_s=f"{clock.seconds():.1f}",
        )

        if do_es and es_count >= patience:
            break

    # IMPORTANT: optimizer naming for aggregation
    # ------------------------------------------------
    # aggregate_results.py groups by (domain, regime, optimizer) taken from metrics.json.
    # If you run CHB with a SAM wrapper, some configs may set optimizer.name = "sam" even
    # when you intend to report SAM (SGD) vs SAM (AdamW) separately.
    #
    # To keep plots/tables consistent, we derive a stable, paper-facing optimizer key:
    # - Prefer cfg["optimizer"]["name"] when it is already specific (e.g. "sam_sgd").
    # - If cfg says "sam", infer the variant from the output directory name when possible.
    #   (This matches common layouts like .../sam_sgd/seed_0/metrics.json.)
    opt_name = str((cfg.get("optimizer") or {}).get("name") or "unknown")
    out_dir_opt = os.path.basename(os.path.dirname(args.out_dir))  # e.g. sam_sgd, sam_adamw
    if opt_name.lower() == "sam" and out_dir_opt.lower().startswith("sam_"):
        opt_name = out_dir_opt

    payload = {
        "domain": "chb",
        "task": "detection",
        "regime": cfg.get("budget", {}).get("regime", "fixed_time"),
        "budget": cfg.get("budget", {}),
        "optimizer": opt_name,
        "seed": int(args.seed),
        "iid_score": float(best_test),
        "robust_scores": {"clean_heldout_patients": float(best_test)},
        "time_to_threshold_s": None if time_to_95 is None else float(time_to_95),
        "time_to_target_s": {"auc@0.90": None if time_to_target is None else float(time_to_target)},
        "curve": {
            "t_s": curve_t_s,
            "train_acc": curve_train_acc,
            "val_auc": curve_val_auc,
            "test_auc": curve_test_auc,
            "train_loss": curve_train_loss,
        },
        "sharpness": None,
        "notes": {
            "best_epoch": int(best_epoch),
            "best_val_auc": float(best_val),
            "wallclock_seconds": float(clock.seconds()),
        },
    }
    logger.save(payload)


if __name__ == "__main__":
    main()