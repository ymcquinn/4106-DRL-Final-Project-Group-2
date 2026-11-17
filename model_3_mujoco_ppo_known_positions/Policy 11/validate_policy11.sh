#!/usr/bin/env bash
# Validate a Policy 11 checkpoint with the same protocol as Policy 10: seeds 17-21, 64 envs, 2700 steps.
#
#   bash validate_policy11.sh policy11.pt                          # camera CNN, image at the parked pose
#   bash validate_policy11.sh policy11.pt --vision gt              # ground-truth positions (Policy 10 inputs)
#   bash validate_policy11.sh policy11.pt --look_pose home         # image taken with the arm at home
#   bash validate_policy11.sh policy11.pt --action_override zeros  # policy output discarded
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY="${PY:-../model_3_mujoco_ppo_known_positions/.venv/bin/python}"
THREADS="${THREADS:-16}"
CKPT="$1"; shift
for seed in 17 18 19 20 21; do
  echo "=== $CKPT $* seed $seed"
  "$PY" liquid_transfer_m3_v11.py --mode play --checkpoint "$CKPT" --num_envs 64 --max_steps 2700 \
    --seed "$seed" --threads "$THREADS" "$@" 2>&1 | grep -E "^\[|^\*\*\*"
done
