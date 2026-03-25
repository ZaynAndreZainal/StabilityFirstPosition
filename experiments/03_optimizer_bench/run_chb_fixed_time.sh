#!/usr/bin/env bash
set -euo pipefail

SRC="bench/src/train_chb.py"
CFG_REGIMES="bench/configs/regimes"
CFG_CHB="bench/configs/chb"
OUT_ROOT="bench/results/raw/chb"

REGIMES=(fixed_time_2h)
OPTS=(sgd_momentum adamw lion sam_sgd sam_adamw)
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

      echo "=== CHB run: regime=${regime} opt=${opt} seed=${seed} ==="
      python3 "${SRC}" \
        --config "${CFG_REGIMES}/${regime}.yaml" \
        --config "${CFG_CHB}/${opt}.yaml" \
        --seed "${seed}" \
        --out_dir "${out_dir}"
    done
  done
done
