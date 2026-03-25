#!/usr/bin/env bash
set -euo pipefail

SRC="bench/src/train_cifar10.py"
CFG_REGIMES="bench/configs/regimes"
CFG_CIFAR="bench/configs/cifar10"
OUT_ROOT="bench/results/raw/cifar10"

REGIMES=(fixed_time_2h fixed_compute)
OPTS=(sgd_momentum adamw lion sam_sgd sam_adamw adam rmsprop sgd)
SEEDS=(0 1 2)

for regime in "${REGIMES[@]}"; do
  for opt in "${OPTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      out_dir="${OUT_ROOT}/${regime}/${opt}/seed_${seed}"
      metrics_file="${out_dir}/metrics.json"

      if [[ -f "${metrics_file}" ]]; then
        echo "SKIP (exists): ${metrics_file}"
        continue
      fi

      echo "=== CIFAR run: regime=${regime} opt=${opt} seed=${seed} ==="
      python3 "${SRC}" \
        --config "${CFG_REGIMES}/${regime}.yaml" \
        --config "${CFG_CIFAR}/${opt}.yaml" \
        --seed "${seed}" \
        --out_dir "${out_dir}"
    done
  done
done
