#!/usr/bin/env bash
set -euo pipefail

# Run from repo root. (If you run this from elsewhere, cd first.)
SCRIPT="bench/src/plot_pareto_tradeoff.py"
SUMMARY_JSON="bench/results/aggregated/summary.json"

# If you leave --out_png at its default in the Python script, it will auto-name
# files based on --domain / --regime and add _downsampled only when downselection
# actually occurred. This script intentionally relies on that behavior.

DOMAINS=("" "cifar10" "chb")
REGIMES=("" "fixed_time" "fixed_compute")
DOWNSAMPLE_FLAGS=("" "--no_downsampling")

for domain in "${DOMAINS[@]}"; do
	for regime in "${REGIMES[@]}"; do
		for down in "${DOWNSAMPLE_FLAGS[@]}"; do
			cmd=(python3 "$SCRIPT" --summary_json "$SUMMARY_JSON")
			if [[ -n "$domain" ]]; then
				cmd+=(--domain "$domain")
			fi
			if [[ -n "$regime" ]]; then
				cmd+=(--regime "$regime")
			fi
			if [[ -n "$down" ]]; then
				# shellcheck disable=SC2206
				cmd+=($down)
			fi

			echo "\n>>> ${cmd[*]}"
			"${cmd[@]}"
		done
	done
done