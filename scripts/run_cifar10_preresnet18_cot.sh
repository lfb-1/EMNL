#!/usr/bin/env bash
set -euo pipefail

DESC=${1:-c10_preresnet18_cot}

for noise in 0.2 0.3 0.4 0.5; do
  python main.py \
    experiment=cifar10 \
    seed=42 \
    desc="${DESC}_r${noise}" \
    experiment.backbone=preresnet18 \
    experiment.r="${noise}" \
    cot=2
done
