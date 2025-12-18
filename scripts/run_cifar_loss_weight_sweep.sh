#!/usr/bin/env bash
set -euo pipefail

# Optional description prefix (default: c10_lw)
DESC_PREFIX=${1:-c10_lw}

SEEDS=(42)
NOISE_RATES=(0.2 0.3 0.4 0.5)
LOSS_FACTORS=(0.2 0.4 0.6 0.8)
GOALS=(pxy)

for seed in "${SEEDS[@]}"; do
  for goal in "${GOALS[@]}"; do
    for noise in "${NOISE_RATES[@]}"; do
      for idx in 0 1 2; do
        for factor in "${LOSS_FACTORS[@]}"; do
          weights=(1.0 1.0 1.0)
          weights[$idx]="${factor}"
          python main.py \
            experiment=cifar10 \
            experiment.r="${noise}" \
            loss_weight="[${weights[0]},${weights[1]},${weights[2]}]" \
            desc="${DESC_PREFIX}_${goal}_r${noise}_lwidx${idx}_${factor}_seed${seed}" \
            optim_goal="${goal}" \
            seed="${seed}"
        done
      done
    done
  done
done
