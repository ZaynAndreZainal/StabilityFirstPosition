from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from tqdm import tqdm
import warnings
import mne

mne.set_log_level("ERROR")

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=r".*Channel names are not unique.*")
warnings.filterwarnings("ignore", message=r".*Number of records from the header does not match the file size.*")
warnings.filterwarnings("ignore", message=r".*Scaling factor is not defined.*")
warnings.filterwarnings("ignore", message=r".*Sampling frequency of the instance is already.*")


# -------------------------
# Summary parsing
# -------------------------

_MNE_DUP = re.compile(r"^(.*?)-\d+$")
_DUP_SUFFIX = re.compile(r"(.*)-\d+$")
RE_FILE = re.compile(r"^\s*File Name:\s*(\S+\.edf)\s*$", re.IGNORECASE)
RE_START = re.compile(r"^\s*Seizure Start Time:\s*(\d+)\s*seconds\s*$", re.IGNORECASE)
RE_END = re.compile(r"^\s*Seizure End Time:\s*(\d+)\s*seconds\s*$", re.IGNORECASE)
_RE_NSZ = re.compile(r"^Number of Seizures in File:\\s*(?P<n>\\d+)\\s*$")
_RE_SSZ = re.compile(r"^Seizure Start Time:\\s*(?P<t>\\d+)\\s*seconds\\s*$")
_RE_ESZ = re.compile(r"^Seizure End Time:\\s*(?P<t>\\d+)\\s*seconds\\s*$")

from collections import Counter

def make_unique(names):
    seen = Counter()
    out = []
    for n in names:
        seen[n] += 1
        if seen[n] == 1:
            out.append(n)
        else:
            out.append(f"{n}__DUP{seen[n]}")
    return out



def parse_chb_summary(summary_path: str) -> Dict[str, List[Tuple[int, int]]]:
    out: Dict[str, List[Tuple[int, int]]] = {}
    cur_edf: Optional[str] = None
    pending_start: Optional[int] = None

    with open(summary_path, "r", errors="ignore") as f:
        for raw_ln in f:
            ln = raw_ln.strip()

            m = RE_FILE.match(ln)
            if m:
                cur_edf = m.group(1)
                out.setdefault(cur_edf, [])
                pending_start = None
                continue

            if cur_edf is None:
                continue

            m = RE_START.match(ln)
            if m:
                pending_start = int(m.group(1))
                continue

            m = RE_END.match(ln)
            if m:
                end_t = int(m.group(1))
                if pending_start is None:
                    # malformed file; skip this end marker
                    continue
                out[cur_edf].append((pending_start, end_t))
                pending_start = None
                continue

    return out


def patient_has_seizure(summary_path: str) -> bool:
  seizures_by_edf = parse_chb_summary(summary_path)
  for intervals in seizures_by_edf.values():
    if len(intervals) > 0:
      return True
  return False

# -------------------------
# Deterministic split + sampling
# -------------------------

def split_patient(patient_id: str, test_pct: int = 20, val_pct: int = 10) -> str:
    """Legacy percent-based split (kept for backwards compatibility).

    Prefer explicit held-out patient lists via --test_patients/--val_patients for reproducible splits.
    """
    h = hashlib.md5(patient_id.encode("utf-8")).hexdigest()
    r = int(h[:8], 16) % 100
    if r < test_pct:
        return "test"
    if r < test_pct + val_pct:
        return "val"
    return "train"


def parse_patient_list(s: str) -> List[str]:
    """Parse comma-separated patient IDs like 'chb05,chb07,chb12'."""
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def stable_key(*parts: Any) -> int:
    s = "|".join(str(p) for p in parts)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def normalize_ch_name(s: str) -> str:
  s = s.strip().upper().replace(" ", "")
  head, sep, tail = s.rpartition("-")
  if sep == "-" and tail.isdigit() and "-" in head:
    s = head
  return s

# -------------------------
# Canonical channels (train-only frequent set)
# -------------------------

def is_valid_eeg_name(n: str) -> bool:
  n = normalize_ch_name(n)
  if n in {"", "-"}:
    return False
  if n.startswith("--"):
    return False
  # CHB bipolar montage usually contains a dash
  if "-" not in n:
    return False
  # reject names like "--0" that still have digits but no real label
  if n.replace("-", "").isdigit():
    return False
  return True


def edf_channel_names_fast(edf_path: str) -> List[str]:
  raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
  chs = [normalize_ch_name(c) for c in raw.ch_names]
  chs = [c for c in chs if is_valid_eeg_name(c)]
  return sorted(set(chs))


def build_canonical_channels(train_patient_dirs: List[str], min_freq: float) -> List[str]:
    """Frequent-channel set across EDF headers in training patients."""
    counts: Dict[str, int] = {}
    total_files = 0

    for pdir in train_patient_dirs:
        edfs = sorted(glob.glob(os.path.join(pdir, "*.edf")))
        for edf_path in edfs:
            chs = edf_channel_names_fast(edf_path)
            for c in set(chs):
                counts[c] = counts.get(c, 0) + 1
            total_files += 1

    if total_files == 0:
        raise RuntimeError("No EDF files found in training patient dirs")

    keep = [c for c, n in counts.items() if (n / total_files) >= min_freq]
    keep = sorted(keep)
    if not keep:
        raise RuntimeError(
            f"Canonical channel set empty. Consider lowering min_freq (current={min_freq})."
        )
    return keep


# -------------------------
# EDF -> preprocessed array
# -------------------------

def load_preprocess_edf(edf_path: str, *, resample_hz: int,
                        bandpass_hz: Tuple[float, float],
                        notch_hz: Optional[float]) -> Tuple[np.ndarray, List[str]]:
  raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

  raw.rename_channels({c: normalize_ch_name(c) for c in raw.ch_names}, allow_duplicates=True)

  # drop junk + duplicates
  valid = [c for c in raw.ch_names if is_valid_eeg_name(c)]
  raw.pick(valid)

  keep = []
  seen = set()
  for ch in raw.ch_names:
    if ch not in seen:
      keep.append(ch)
      seen.add(ch)
  raw.pick(keep)

  if resample_hz is not None:
    raw.resample(resample_hz)

  l_freq, h_freq = bandpass_hz
  raw.filter(l_freq=float(l_freq), h_freq=float(h_freq), verbose=False)

  if notch_hz is not None:
    raw.notch_filter(freqs=[float(notch_hz)], verbose=False)

  X = raw.get_data().astype(np.float32)
  return X, list(raw.ch_names)


def project_to_canonical_with_mask(
    X: np.ndarray,
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
            Xc[j] = X[idx]
            mask[j] = 1.0
    return Xc, mask


def concat_mask_as_channels(Xc: np.ndarray, mask: np.ndarray) -> np.ndarray:
    C, T = Xc.shape
    M = np.repeat(mask[:, None], T, axis=1).astype(np.float32)
    return np.concatenate([Xc, M], axis=0)  # [2C, T]


# -------------------------
# Windowing + labeling
# -------------------------

def overlaps_any_interval(t0: float, t1: float, intervals: List[Tuple[int, int]]) -> int:
    for s, e in intervals:
        if t0 < e and t1 > s:
            return 1
    return 0


def make_windows(
    X: np.ndarray,
    intervals: List[Tuple[int, int]],
    *,
    fs: int,
    window_seconds: float,
    stride_seconds: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Xw, y, t0_sec) where Xw is [N, C, T_win]."""
    C, T = X.shape
    win = int(round(window_seconds * fs))
    stride = int(round(stride_seconds * fs))

    starts = list(range(0, max(T - win + 1, 0), stride))

    Xw = np.zeros((len(starts), C, win), dtype=np.float32)
    y = np.zeros((len(starts),), dtype=np.int64)
    t0s = np.zeros((len(starts),), dtype=np.float32)

    for i, s0 in enumerate(starts):
        s1 = s0 + win
        Xw[i] = X[:, s0:s1]
        t0 = s0 / fs
        t1 = s1 / fs
        y[i] = overlaps_any_interval(t0, t1, intervals)
        t0s[i] = t0

    return Xw, y, t0s


# -------------------------
# Negative cap (10:1) deterministic
# -------------------------

def cap_negatives_deterministic(
    Xw: np.ndarray,
    y: np.ndarray,
    meta_keys: List[int],
    neg_to_pos: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if len(pos_idx) == 0:
        # If no positives in this unit, keep a small deterministic subset of negatives
        keep = min(len(neg_idx), 1024)
        neg_sorted = sorted(neg_idx.tolist(), key=lambda i: meta_keys[i])
        sel = np.array(neg_sorted[:keep], dtype=int)
        return Xw[sel], y[sel]

    max_neg = min(len(neg_idx), neg_to_pos * len(pos_idx))
    neg_sorted = sorted(neg_idx.tolist(), key=lambda i: meta_keys[i])
    neg_sel = np.array(neg_sorted[:max_neg], dtype=int)

    sel = np.concatenate([pos_idx, neg_sel])
    # Keep deterministic ordering as well
    sel = np.array(sorted(sel.tolist(), key=lambda i: meta_keys[i]), dtype=int)

    return Xw[sel], y[sel]


# -------------------------
# Shard writing
# -------------------------

def shard_buffer_init():
  return {"X": [], "y": [], "n": 0, "shard_idx": 0}


def shard_buffer_add(buf, Xp: np.ndarray, yp: np.ndarray, out_dir: str, shard_size: int):
  """
  Append arrays to buffer and flush full shards to disk.
  Keeps remainder in buffer.
  """
  os.makedirs(out_dir, exist_ok=True)

  buf["X"].append(Xp)
  buf["y"].append(yp)
  buf["n"] += int(Xp.shape[0])

  while buf["n"] >= shard_size:
    Xcat = np.concatenate(buf["X"], axis=0)
    ycat = np.concatenate(buf["y"], axis=0)

    Xwrite = Xcat[:shard_size]
    ywrite = ycat[:shard_size]

    shard_path = os.path.join(out_dir, f"shard_{buf['shard_idx']:03d}.npz")
    np.savez_compressed(shard_path, X=Xwrite, y=ywrite)
    buf["shard_idx"] += 1

    # remainder back into buffer
    Xrem = Xcat[shard_size:]
    yrem = ycat[shard_size:]

    buf["X"] = [Xrem] if Xrem.shape[0] else []
    buf["y"] = [yrem] if yrem.shape[0] else []
    buf["n"] = int(Xrem.shape[0])


def shard_buffer_flush(buf, out_dir: str):
  if buf["n"] == 0:
    return
  os.makedirs(out_dir, exist_ok=True)
  Xcat = np.concatenate(buf["X"], axis=0)
  ycat = np.concatenate(buf["y"], axis=0)
  shard_path = os.path.join(out_dir, f"shard_{buf['shard_idx']:03d}.npz")
  np.savez_compressed(shard_path, X=Xcat, y=ycat)
  buf["shard_idx"] += 1
  buf["X"], buf["y"], buf["n"] = [], [], 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf_root", type=str, required=True)
    ap.add_argument("--cache_root", type=str, default="/mnt/data12_16T/andre/cache")
    ap.add_argument("--cache_name", type=str, default="chb_detection_v1")

    ap.add_argument("--test_pct", type=int, default=20)
    ap.add_argument("--val_pct", type=int, default=10)

    # Recommended: explicit, reproducible held-out patient lists.
    # If provided, they override the percent-based hash split for seizure-positive patients.
    ap.add_argument(
        "--test_patients",
        type=str,
        default="chb05,chb07,chb12,chb18,chb24",
        help="Comma-separated patient IDs to hold out for test (e.g. chb05,chb07,...)",
    )
    ap.add_argument(
        "--val_patients",
        type=str,
        default="chb03,chb14,chb21",
        help="Comma-separated patient IDs to hold out for validation (e.g. chb03,chb14,...)",
    )

    ap.add_argument("--min_freq", type=float, default=0.9)
    ap.add_argument("--neg_to_pos", type=int, default=10)
    ap.add_argument("--shard_size", type=int, default=4096)

    ap.add_argument("--resample_hz", type=int, default=256)
    ap.add_argument("--bandpass_low", type=float, default=0.5)
    ap.add_argument("--bandpass_high", type=float, default=40.0)
    ap.add_argument("--notch_hz", type=float, default=60.0)

    ap.add_argument("--window_seconds", type=float, default=5.0)
    ap.add_argument("--stride_seconds", type=float, default=2.5)
    args = ap.parse_args()

    out_root = os.path.join(args.cache_root, args.cache_name)
    meta_dir = os.path.join(out_root, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    # Discover patients
    patient_dirs = sorted([p for p in glob.glob(os.path.join(args.edf_root, "chb*")) if os.path.isdir(p)])
    if not patient_dirs:
        raise FileNotFoundError(f"No patient dirs found under {args.edf_root}")

    pos_patient_dirs = []
    neg_patient_dirs = []

    for pdir in patient_dirs:
      pid = os.path.basename(pdir)
      summary_path = os.path.join(pdir, f"{pid}-summary.txt")
      if not os.path.exists(summary_path):
        continue
      if patient_has_seizure(summary_path):
        pos_patient_dirs.append(pdir)
      else:
        neg_patient_dirs.append(pdir)

    pos_patient_dirs = sorted(pos_patient_dirs)
    neg_patient_dirs = sorted(neg_patient_dirs)

    train_dirs, val_dirs, test_dirs = [], [], []

    # -------------------------
    # Best-practice explicit split
    # -------------------------
    test_ids = set(parse_patient_list(args.test_patients))
    val_ids = set(parse_patient_list(args.val_patients))

    if test_ids & val_ids:
      overlap = sorted(test_ids & val_ids)
      raise ValueError(f"val/test patient lists overlap: {overlap}")

    for pdir in pos_patient_dirs:
      pid = os.path.basename(pdir)

      if pid in test_ids:
        test_dirs.append(pdir)
      elif pid in val_ids:
        val_dirs.append(pdir)
      else:
        # fallback for the remaining seizure-positive patients
        # (keeps older behavior if you want randomization without editing lists)
        bucket = split_patient(pid, test_pct=args.test_pct, val_pct=args.val_pct)
        if bucket == "test":
          test_dirs.append(pdir)
        elif bucket == "val":
          val_dirs.append(pdir)
        else:
          train_dirs.append(pdir)

    # Put all seizure-negative patients into train (so val/test keep positives)
    train_dirs.extend(neg_patient_dirs)
    train_dirs = sorted(train_dirs)

    if len(val_dirs) == 0 or len(test_dirs) == 0:
      raise RuntimeError(
        f"Split produced val={len(val_dirs)} test={len(test_dirs)}. "
        "Check --test_patients/--val_patients or adjust --test_pct/--val_pct."
      )

    split_payload = {
        "test_pct": args.test_pct,
        "val_pct": args.val_pct,
        "train_patients": [os.path.basename(p) for p in train_dirs],
        "val_patients": [os.path.basename(p) for p in val_dirs],
        "test_patients": [os.path.basename(p) for p in test_dirs],
    }
    with open(os.path.join(meta_dir, "split.json"), "w") as f:
        json.dump(split_payload, f, indent=2)

    # Canonical channels from TRAIN only
    canonical = build_canonical_channels(train_dirs, min_freq=args.min_freq)
    with open(os.path.join(meta_dir, "canonical_channels.json"), "w") as f:
        json.dump({"min_freq": args.min_freq, "channels": canonical}, f, indent=2)

    # Accumulators for normalization (train only, signal channels only)
    sig_sum = np.zeros((len(canonical),), dtype=np.float64)
    sig_sumsq = np.zeros((len(canonical),), dtype=np.float64)
    sig_count = np.zeros((len(canonical),), dtype=np.float64)

    def process_split(split_name: str, dirs: List[str], accumulate_norm: bool):
      out_dir = os.path.join(out_root, split_name)
      buf = shard_buffer_init()

      n_patients_used = 0
      n_windows_written_est = 0

      for pdir in tqdm(dirs, desc=f"{split_name}: patients", unit="patient"):
        pid = os.path.basename(pdir)
        summary_path = os.path.join(pdir, f"{pid}-summary.txt")
        if not os.path.exists(summary_path):
          continue

        seizures_by_edf = parse_chb_summary(summary_path)

        # IMPORTANT: iterate EDFs on disk (robust to filename mismatches / missing summary entries)
        edfs_on_disk = sorted(glob.glob(os.path.join(pdir, "*.edf")))
        if not edfs_on_disk:
          continue

        # per-patient pools (so neg cap is per patient)
        Xp_chunks = []
        yp_chunks = []
        keys: List[int] = []

        for edf_path in edfs_on_disk:
          edf_name = os.path.basename(edf_path)
          intervals = seizures_by_edf.get(edf_name, [])  # if missing in summary -> assume no seizures

          X_raw, ch_names = load_preprocess_edf(
            edf_path,
            resample_hz=args.resample_hz,
            bandpass_hz=(args.bandpass_low, args.bandpass_high),
            notch_hz=args.notch_hz,
          )

          Xc, mask = project_to_canonical_with_mask(X_raw, ch_names, canonical)

          Xw_sig, y, t0s = make_windows(
            Xc,
            intervals,
            fs=args.resample_hz,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
          )

          if Xw_sig.shape[0] == 0:
            continue

          # stable keys for deterministic selection and ordering
          for t0 in t0s:
            keys.append(stable_key(pid, edf_name, float(t0)))

          # concat mask channels AFTER windowing
          T = Xw_sig.shape[2]
          M = np.repeat(mask[None, :, None], Xw_sig.shape[0], axis=0)
          M = np.repeat(M, T, axis=2).astype(np.float32)  # [N, C, T]
          Xw = np.concatenate([Xw_sig, M], axis=1).astype(np.float32)  # [N, 2C, T]

          Xp_chunks.append(Xw)
          yp_chunks.append(y)

          # accumulate norm stats using TRAIN ONLY, PRESENT channels only
          if accumulate_norm:
            present = mask.astype(bool)
            if present.any():
              Xp_sig = Xw_sig[:, present, :].astype(np.float64)
              sig_sum[present] += Xp_sig.sum(axis=(0, 2))
              sig_sumsq[present] += (Xp_sig ** 2).sum(axis=(0, 2))
              sig_count[present] += float(Xp_sig.shape[0] * Xp_sig.shape[2])

        if not Xp_chunks:
          continue

        Xp = np.concatenate(Xp_chunks, axis=0)
        yp = np.concatenate(yp_chunks, axis=0)

        # Deterministic neg cap: pass Python int keys directly (avoid int64 overflow)
        Xp, yp = cap_negatives_deterministic(Xp, yp, keys, neg_to_pos=args.neg_to_pos)

        shard_buffer_add(buf, Xp, yp, out_dir=out_dir, shard_size=args.shard_size)

        n_patients_used += 1
        n_windows_written_est += int(Xp.shape[0])

      shard_buffer_flush(buf, out_dir=out_dir)

      if n_patients_used == 0:
        raise RuntimeError(f"No patients produced windows for split={split_name}")

      print(f"[{split_name}] patients_used={n_patients_used} approx_windows_kept={n_windows_written_est}")

    # Build cache splits
    process_split("train", train_dirs, accumulate_norm=True)
    process_split("val", val_dirs, accumulate_norm=False)
    process_split("test", test_dirs, accumulate_norm=False)

    # Finalize norm stats
    mean = (sig_sum / np.maximum(sig_count, 1.0)).astype(np.float32)
    var = (sig_sumsq / np.maximum(sig_count, 1.0) - mean.astype(np.float64) ** 2)
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var).astype(np.float32)

    np.savez_compressed(os.path.join(meta_dir, "norm_stats.npz"), mean=mean, std=std)

    cache_cfg = {
        "edf_root": args.edf_root,
        "cache_root": args.cache_root,
        "cache_name": args.cache_name,
        "test_pct": args.test_pct,
        "val_pct": args.val_pct,
        "min_freq": args.min_freq,
        "neg_to_pos": args.neg_to_pos,
        "shard_size": args.shard_size,
        "resample_hz": args.resample_hz,
        "bandpass_hz": [args.bandpass_low, args.bandpass_high],
        "notch_hz": args.notch_hz,
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
    }
    with open(os.path.join(meta_dir, "cache_config.json"), "w") as f:
        json.dump(cache_cfg, f, indent=2)

    print(f"Cache written to: {out_root}")


if __name__ == "__main__":
    main()