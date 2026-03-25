# Optimizer Benchmark (CIFAR-10 + CHB-MIT)

A minimal, end-to-end benchmark suite for comparing optimizers across:
- Vision: CIFAR-10 (classification)
- EEG: CHB-MIT (seizure detection)

The pipeline produces per-run metrics.json files, aggregates them across seeds, and generates:
- A NeurIPS-style LaTeX results table
- A Pareto efficiency–performance plot
- Accuracy/AUC vs wallclock time learning curves

---

## Repository layout (expected)

```text
bench/
├── configs/                # YAML hyperparameters for each optimizer
│   └── cifar10/            # (base, sgd, adamw, lion, sam, etc.)
├── src/                    # Core logic
│   ├── train_cifar10.py    # Vision training entry point
│   ├── train_chb.py        # EEG training entry point
│   ├── eval_*.py           # Specialized evaluation scripts
│   ├── aggregate_results.py # Multi-seed statistics
│   └── plot_*.py           # Visualization suite
└── results/                
    ├── raw/                # Written by training scripts (metrics.json)
    └── aggregated/         # Written by analysis scripts
```

---

## Setup

### 1) Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 2) Install dependencies

Minimum (typical):

```bash
pip install numpy matplotlib pyyaml tqdm scikit-learn
pip install torch torchvision
```

Notes:
- train_cifar10.py requires torchvision.
- train_chb.py expects your CHB cache pipeline (see below).

---

## Running experiments

### CIFAR-10 training (example)

```bash
python3 bench/src/train_cifar10.py \
  --config bench/configs/cifar10/base.yaml \
  --config bench/configs/cifar10/sam_sgd.yaml \
  --seed 0 \
  --out_dir bench/results/raw/cifar10/fixed_time_2h/sam_sgd/seed0
```

Repeat over multiple seeds (example):

```bash
for seed in 0 1 2; do
  python3 bench/src/train_cifar10.py \
    --config bench/configs/cifar10/base.yaml \
    --config bench/configs/cifar10/adamw.yaml \
    --seed ${seed} \
    --out_dir bench/results/raw/cifar10/fixed_time_2h/adamw/seed${seed}
done
```

### CHB-MIT training

train_chb.py expects a cached window dataset (npz shards). Two common workflows:

1) Preprocess once to a cache directory (recommended):
   Use bench/src/preprocess_chb_cache.py to generate a cache under:
   /path/to/cache_root/chb_detection_v1/{train,val,test}/shard_*.npz

2) Train from that cache:

```bash
python3 bench/src/train_chb.py \
  --config path/to/your/chb_base.yaml \
  --seed 0 \
  --out_dir bench/results/raw/chb/fixed_time_2h/adamw/seed0
```

Make sure your CHB config points to the correct cache root/name via:
- chb.cache.root
- chb.cache.name

---

## Outputs (what each run writes)

Each training run writes:
- bench/results/raw/**/metrics.json

Key fields:
- domain: cifar10 or chb
- regime: fixed_time / fixed_compute
- optimizer: string key used downstream (keep consistent)
- iid_score: best IID metric in-run (accuracy for CIFAR, AUC for CHB)
- time_to_target_s: seconds to hit the fixed target metric (if configured)
- curve: arrays for learning-curve plots

---

## Aggregate results across seeds

```bash
python3 bench/src/aggregate_results.py \
  --results_root bench/results/raw \
  --out_json bench/results/aggregated/summary.json
```

---

## Generate the LaTeX table

```bash
python3 bench/src/export_latex_table.py \
  --summary_json bench/results/aggregated/summary.json \
  > bench/results/aggregated/results_table.tex
```
---

## Generate Pareto trade-off figure

```bash
python3 bench/src/plot_pareto_tradeoff.py \
  --summary_json bench/results/aggregated/summary.json
```

---

## Plot accuracy/AUC over wallclock time (learning curves)

```bash
python3 bench/src/plot_accuracy_over_time.py \
  --results_root bench/results/raw
```
---

## Reproducibility notes

- Always record:
  - seed
  - optimizer config file(s)
  - budget regime + wallclock budget
- Keep output directories consistent: downstream scripts infer keys from metrics.json and paths.

---

## Troubleshooting

- No plots produced / empty curves: confirm metrics.json includes curve and has 2+ points.
- CHB dataset errors: confirm your cache directory exists and contains shard_*.npz files in train/, val/, test/.
- GPU/torch issues: pin a known-good torch/torchvision build that matches your CUDA driver.
