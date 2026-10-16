# Model 3 (MuJoCo): PPO with known container positions, random beaker and dish

Forked from `../model_2_mujoco_teacher_student_random_dish_positions/`. Instead of the overhead camera, the policy
is told the true beaker and dish positions (the numbers model 2's teacher saw) and is trained directly with PPO.
Both containers are placed at random every episode.

| File | What it is |
|---|---|
| `liquid_transfer_m3.py` | The environment, the PPO agent, and all modes |
| `liquid_transfer_scene.xml`, `panda_pipette.xml`, `franka_emika_panda/` | Scene and robot, identical to model 2 (MuJoCo) |
| `policy.pt` | The trained policy, PPO iteration 1999 |
| `logs/rsl_rl/liquid_transfer_m3/` | TensorBoard log and checkpoints of the training run |

## Task

Metres, robot frame: x forward from the base, y to the robot's left.

- Beaker: nominal (0.40, -0.20) + U(-0.20, 0.20) in x and y.
- Dish: nominal (0.40, 0.20) + U(-0.20, 0.20) in x and y.
- Both are redrawn when either centre is closer than 0.29 m to the base (unreachable with a vertical pipette) or
  the two centres are closer than 0.18 m (they would overlap).
- Order: 2 to 10 ml; success = dish within 0.3 ml of the order. Everything else (liquid model, reward v4,
  terminations, held-target IK action at 1 cm/step, 30 Hz, 15 s episodes) is model 2.
- Workspace box widened to y >= -0.45 so a beaker at y = -0.40 is reachable.

Observations: `policy` = tip position (3), order / pipette / dish ml (3), last action (3), beaker xy (2),
dish xy (2) = 13; `critic` adds the 9 joint velocities (22). Agent: PPO, MLP 256-128, the model 2 teacher's
hyperparameters.

## Results

Training: 128 envs, 2000 iterations, 25 minutes on the CPU (0.75 s / iteration), no crash. Training-time
success (with exploration noise, 100-iteration windows): 15% at iteration 200-300, 44% at 400-500, 76% at
700-800, 91% at 1000-1100, 98-99% from 1500. Zero overfills and zero table contacts over the whole run.

Deterministic evaluation (noise off, seed 7):

| Layout | Success | Episodes | Mean length |
|---|---|---|---|
| Random beaker and dish (the training distribution) | **96.1%** | 369 / 384 | 142 steps |
| Beaker fixed at nominal, random dish (model 2's task) | 99.3% | 413 / 416 | 134 steps |
| Far corners: beaker (0.60, -0.40), dish (0.60, 0.40) | 96.2% | 51 / 53 | 223 steps |
| Near corners: beaker (0.20, -0.40), dish (0.20, 0.40) | 86.5% | 45 / 52 | 232 steps |
| Scripted controller, random beaker and dish | 100% | 63 / 63 | 207 steps |

Success by container distance from nominal (random layout): dish 100 / 98.5 / 99.0 / 93.4 / 93.5% for the
0-5 / 5-10 / 10-15 / 15-20 / 20+ cm bins; beaker 100 / 100 / 99.1 / 96.8 / 85.3%. All failures are time-outs:
the weak spot is a beaker far from nominal (20+ cm, the corners near the base), the same reachability edge
model 2 had for the dish.

For comparison, model 2's teacher (same observation, beaker fixed) scored 97.8% in Isaac Lab and 98.0% in MuJoCo.

## Running

Use the Isaac Lab venv (mujoco, torch, rsl_rl, tensorboard are installed there). Run from this folder.

```bash
cd ~/dev/opencode_work/liquid_transfer/model_3_mujoco_ppo_known_positions
PY=../IsaacLab/.venv/bin/python

# Success numbers (headless)
$PY liquid_transfer_m3.py --mode play --num_envs 32 --max_steps 1800 --seed 7 --checkpoint policy.pt

# Watch one episode (random beaker and dish; change --seed for another layout and order)
$PY liquid_transfer_m3.py --mode play --num_envs 1 --max_steps 1000 --hold --viz --checkpoint policy.pt

# Chosen positions: beaker at (0.40 + dx, -0.20 + dy), dish at (0.40 + dx, 0.20 + dy); use '=' so negatives parse
$PY liquid_transfer_m3.py --mode play --num_envs 1 --max_steps 1000 --hold --viz \
  --beaker_offset=0.2,-0.2 --dish_offset=0.2,0.2 --checkpoint policy.pt

# Scripted baseline and reachability probe
$PY liquid_transfer_m3.py --mode scripted --num_envs 16 --max_steps 900
$PY liquid_transfer_m3.py --mode probe --num_envs 8

# Train from scratch (about 25 minutes)
$PY liquid_transfer_m3.py --mode train --num_envs 128 --max_iterations 2000 --run_name my_run
```

`--hold` runs one episode without auto-reset and keeps the window open on the final frame; close the window when
done. The terminal prints `target / pipette / dish` ml every 50 steps and a final line with the outcome and both
container offsets. Other flags: `--dish_range`, `--beaker_range` (default 0.20), `--episode_length_s`, `--seed`,
`--threads`, `--device`.
