# Policy 11 — Results Summary

Policy 11 does Policy 10's liquid-transfer task without being told where the beaker and dish are.
An overhead camera image goes through an AlexNet convolutional network, which outputs the two
container positions; the policy uses those instead of the ground-truth positions Policy 10 was given.

- `policy11.pt` — the complete policy (perception network + actor).
- `perception.py` — the AlexNet locator: model, training, evaluation.
- `liquid_transfer_m3_v11.py` — environment, training and evaluation (Policy 10's script plus the camera).
- Nothing in `Policy 10/` was changed.

## Result

Same protocol as Policy 10: seeds 17–21, random container placement ±17.5 cm per axis, 20 s episodes,
64 envs × 2700 steps per seed.

| Condition | Success | Safety failures |
|---|---|---|
| **Policy 11** (positions from the camera) | **2,742 / 2,791 (98.2%)** | 0 |
| Policy 10 (ground-truth positions), re-run with `--vision gt` | 2,732 / 2,785 (98.1%) | 0 |
| Policy 11, image taken with the arm at home instead of parked | 2,721 / 2,782 (97.8%) | 0 |
| Policy 11, policy output replaced by zeros (seed 17) | 0 / 256 (0%) | 0 |

- Per seed: 97.5 / 98.6 / 98.4 / 98.6 / 98.2 %
- Mean episode length: ~293 steps (~9.8 s at 30 Hz), the same as Policy 10.
- Failures by stage: 38 approach (0), 10 transit (3), 1 lift-out (5). All are time-outs, the same
  pattern as Policy 10.
- Perception error during these runs: beaker mean 0.6 mm, max 2.6 mm; dish mean 0.5 mm, max 1.9 mm.
  The error in failed episodes is no larger than in successful ones, so the remaining failures come
  from the motion policy, not from the camera.

Success by distance from the nominal position:

| Offset from nominal | 0–5 cm | 5–10 cm | 10–15 cm | 15–20 cm | 20+ cm |
|---|---|---|---|---|---|
| Beaker | 97.2% | 98.6% | 98.8% | 97.4% | 100.0% |
| Dish | 98.4% | 98.8% | 98.6% | 98.0% | 95.8% |

## What was trained and what was not

- **The perception network is new** and was trained from scratch for this policy.
- **The actor weights are Policy 10's, unchanged.** With sub-millimetre position estimates the Policy 10
  actor already reaches Policy 10's success rate, so no PPO fine-tuning was run. The fine-tune path
  works (a 2-iteration run with the camera in the loop was checked) and is described below.

## How it works

1. At the start of an episode the overhead camera (`overhead` in `liquid_transfer_scene.xml`: 1.0 m
   above the table, looking straight down) takes one 256 × 192 RGB image.
2. The AlexNet network outputs four numbers: beaker x, y and dish x, y in the robot frame.
3. The policy observation (24 numbers, same layout as Policy 10) is built from these estimates. The
   estimate is held for the episode, because the containers do not move.
4. Ground-truth container positions are used only by the simulator (rewards, stage transitions,
   success checks), by the critic during training, and to report the perception error.

### Where the arm is when the image is taken

- The image is taken with the arm in a parked pose (leaned back, `PARK_QPOS` in `perception.py`).
  From there it hides neither container anywhere in the placement range.
- With the arm at home it hides more than half of the beaker in about 3% of layouts.
- The simulator renders the parked-pose image and starts the episode at home; the move from parked to
  home is not simulated.
- `--look_pose home` takes the image at the nominal home pose instead (97.8% above). The network was
  trained on both poses.

### Perception network

- AlexNet's five convolutional layers (torchvision layout), then a 256-unit layer and a 4-output
  layer in place of AlexNet's classifier. About 4.8 M parameters, 19 MB.
- Trained from scratch, supervised, on images rendered from the scene: 6,000 steps × 64 fresh images,
  with brightness, colour and noise augmentation. 44 minutes on CPU. Log: `perception_train.log`.
- Training placements cover ±20 cm, slightly wider than the task's ±17.5 cm.

Held-out error (5,000 images, task placement range, no augmentation):

| Arm pose in image | Container | Mean | 95th pct | Max | Over 10 mm |
|---|---|---|---|---|---|
| Parked | Beaker | 0.61 mm | 1.21 mm | 1.85 mm | 0% |
| Parked | Dish | 0.53 mm | 1.03 mm | 2.19 mm | 0% |
| Home, ±0.1 rad joint noise | Beaker | 1.41 mm | 1.98 mm | 120 mm | 1.36% |
| Home, ±0.1 rad joint noise | Dish | 0.66 mm | 1.36 mm | 24.7 mm | 0.08% |

For scale, the task requires the tip within 20 mm of the container axis.

## Limits

- Everything in `Policy 10/Physical_setup.md` still applies (placement range, workspace clip, beaker on
  the −y side and dish on the +y side).
- The network has only seen this simulated scene: one camera pose, one floor, one beaker and one dish,
  with mild brightness and noise variation. It is not expected to work on a real camera without
  retraining on real or more heavily randomized images.
- One image per episode. A container that moves after the snapshot is not tracked.
- `--viz` (viewer and camera renderer together) was not tested in this headless session.

## Commands

Run from `Policy 11/`. `PY=../model_3_mujoco_ppo_known_positions/.venv/bin/python` (no new packages needed).

```bash
# Validate (seeds 17-21); extra flags are passed through
bash validate_policy11.sh policy11.pt
bash validate_policy11.sh policy11.pt --vision gt          # ground-truth positions
bash validate_policy11.sh policy11.pt --look_pose home     # image taken at home

# One seed, or watch one episode
$PY liquid_transfer_m3_v11.py --mode play --checkpoint policy11.pt --num_envs 64 --max_steps 2700 --seed 17
$PY liquid_transfer_m3_v11.py --mode play --checkpoint policy11.pt --num_envs 1 --max_steps 900 --seed 17 --viz --trace

# Perception only
$PY perception.py --mode eval --checkpoint perception.pt
$PY perception.py --mode preview        # writes camera_preview_park.png / camera_preview_home.png
$PY perception.py --mode train          # retrain (about 45 min on CPU), writes perception.pt
```

`--beaker_offset`, `--dish_offset` and `--action_override` work as in Policy 10.

### Fine-tuning the actor with the camera in the loop

Not needed for the result above. If the perception changes (a new camera, a noisier network), fine-tune
from the current policy and validate before promoting, as for Policy 10:

```bash
$PY liquid_transfer_m3_v11.py --mode train --phase finetune --checkpoint policy11.pt \
  --max_iterations 100 --num_envs 128 --threads 16 --warm_start_std 0 --run_name finetune_r1
bash validate_policy11.sh checkpoints/finetune.pt
```

The saved checkpoint includes the perception network. Pass `--perception <file>` to use a different one.

## Logs

- `validation.log` — Policy 11, seeds 17–21.
- `validation_gt.log` — ground-truth positions (reproduces Policy 10's per-seed numbers).
- `validation_look_home.log` — image taken at home.
- `validation_zero_action.log` — zero-action ablation, seed 17.
