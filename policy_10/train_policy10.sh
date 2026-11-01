#!/usr/bin/env bash
# Train Policy 10 through the staged curriculum. Each phase warm-starts from the previous phase.
#
#   bash train_policy10.sh                 # all phases
#   START=chain bash train_policy10.sh     # resume from a phase (uses checkpoints/<previous>.pt)
#
# Phases and PPO iterations (128 envs x 32 steps = 4096 env steps per iteration):
#   approach        stage 0 only: reach and center above the beaker
#   aspirate_lift   stages 1-2: descend, aspirate 2 mL, lift clear (+ approach replay)
#   transit         stage 3: carry to above the dish (+ replay)
#   dispense_lift   stages 4-5: descend, dispense, lift clear (+ replay)
#   chain           complete task from every start stage (reverse curriculum)
#   end_to_end      complete task, mostly from home
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY="${PY:-../model_3_mujoco_ppo_known_positions/.venv/bin/python}"
NUM_ENVS="${NUM_ENVS:-128}"
THREADS="${THREADS:-16}"
START="${START:-approach}"

PHASES=(approach:150 aspirate_lift:250 transit:300 dispense_lift:350 chain:400 end_to_end:400)

prev=""
started=0
for spec in "${PHASES[@]}"; do
  phase="${spec%%:*}"; iters="${spec##*:}"
  if [[ "$phase" == "$START" ]]; then started=1; fi
  if [[ $started == 0 ]]; then prev="checkpoints/$phase.pt"; continue; fi
  echo "=== phase $phase ($iters iterations) ${prev:+from $prev}"
  "$PY" liquid_transfer_m3_v10.py --mode train --phase "$phase" --max_iterations "$iters" \
    --num_envs "$NUM_ENVS" --threads "$THREADS" ${prev:+--checkpoint "$prev"}
  prev="checkpoints/$phase.pt"
done
cp checkpoints/end_to_end.pt policy10.pt
echo "=== done: policy10.pt"
