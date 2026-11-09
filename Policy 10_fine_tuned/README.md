# Policy 10 — Results Summary

Policy 10 (`liquid_transfer_m3_v10.py`, checkpoint `policy10.pt`) is a learned PPO policy for the
six-stage liquid-transfer task (approach beaker → aspirate → lift → transit → dispense → lift out of dish).
Unlike v6, the policy's actions actually drive the robot: no hand-written controller overrides them.

## Training

- Curriculum adapted from v6 (`liquid_transfer_m3_v6_trained.py`), ending with `end_to_end` (400 iters)
  followed by a `finetune` phase.
- Fine-tune fixed sideways drift off far dishes during the stage-5 lift (off-axis penalty now covers the
  whole stage-5 lift).
  - Round 1 (150 iters): stage-5 failures 58 → 0, but approach (stage 0) failures rose 9 → 70 (97.0% overall).
  - Round 2 (100 iters, with the approach-only skill `(0, 0)` added back to the mix): 98.1% overall.
- Pre-fine-tune checkpoint and logs are kept in `old_run/`.

## Validation (current `policy10.pt`)

Seeds 17–21, random container placement ±17.5 cm per axis, 20 s episodes.

| Condition | Success | Safety failures |
|---|---|---|
| Policy | **2,732 / 2,785 (98.1%)** | 0 |
| Policy output replaced by zeros (ablation) | 0 / 1,280 (0%) | 0 |

The zero-action ablation shows that success comes from the learned policy, not a scripted controller.

- Per seed: 98.4 / 96.9 / 98.2 / 98.4 / 98.6 %
- Mean episode length: ~293 steps (~9.8 s at 30 Hz)
- Failures by stage: 39 approach (0), 1 aspirate (1), 10 transit (3), 3 lift-out (5)

Success by distance from the nominal position:

| Offset from nominal | 0–5 cm | 5–10 cm | 10–15 cm | 15–20 cm | 20+ cm |
|---|---|---|---|---|---|
| Beaker | 98.5% | 97.2% | 98.3% | 98.1% | 99.5% |
| Dish | 98.9% | 98.5% | 98.5% | 97.8% | 95.6% |

## Placement limits

Inherited from v6:

- Nominal positions: beaker (0.40, −0.20, 0.06), dish (0.40, 0.20, 0.01).
- Trained random range: ±0.175 m per axis. This is the range v6 (`model_999.pt`) was validated at (2,004 / 2,004
  on seeds 17–21), not v6's code default of ±0.20 m. v6 was not validated past ±17.5 cm.
- Each container must be ≥ 0.29 m from the robot base, and the two must be ≥ 0.18 m apart. Invalid layouts
  are rejected.
- Tip workspace clip: x 0.15–0.65 m, y ±0.45 m, z 0.03–0.45 m.

Measured physical reach (tip pointing down, horizontal from the base): 0.810 m at z = 0.04, 0.805 m at
z = 0.12, 0.790 m at z = 0.20. Reach is the same in every direction. In practice the workspace clip,
not the arm, sets the limit.

## Layout tests at 0.30 m from the base

Seed 17, 16 environments, 1800 steps.

| Layout | Beaker | Dish | Separation | In trained range | Success |
|---|---|---|---|---|---|
| A | (0.225, −0.20) | (0.225, 0.20) | 0.40 m | Yes (corner) | 77 / 85 (90.6%) |
| B | (0.15, −0.26) | (0.15, 0.26) | 0.52 m | No | 33 / 56 (58.9%) |

- In both layouts, every failure was a stall before reaching the beaker (stage 0).
- B is the widest separation the workspace allows at 0.30 m. The geometric maximum is 0.60 m (at x = 0),
  but that lies outside the x ≥ 0.15 m workspace.

## Commands

Run from `Policy 10/`:

```bash
# Layout A (watch one episode)
../model_3_mujoco_ppo_known_positions/.venv/bin/python liquid_transfer_m3_v10.py --mode play \
  --checkpoint policy10.pt --num_envs 1 --max_steps 900 --seed 17 \
  --beaker_offset -0.175,0 --dish_offset -0.175,0 --viz --trace

# Layout B (watch one episode)
../model_3_mujoco_ppo_known_positions/.venv/bin/python liquid_transfer_m3_v10.py --mode play \
  --checkpoint policy10.pt --num_envs 1 --max_steps 900 --seed 17 \
  --beaker_offset -0.25,-0.06 --dish_offset -0.25,0.06 --viz --trace
```

To measure a success rate instead of watching, use `--num_envs 16 --max_steps 1800` and drop `--viz --trace`.
Add `--action_override zeros` to run the zero-action ablation.

## Training further

Run from `Policy 10/`. Further training is another fine-tune round that warm-starts from `policy10.pt`.
The last round (100 iterations, 128 envs, 16 threads) took about 2.5 minutes on CPU.

1. Back up the current policy. Training always writes to `checkpoints/<phase>.pt`, so a new `finetune`
   round overwrites `checkpoints/finetune.pt`.

   ```bash
   cp policy10.pt old_run/policy10_98.1.pt
   ```

2. Run a fine-tune round.

   ```bash
   ../model_3_mujoco_ppo_known_positions/.venv/bin/python liquid_transfer_m3_v10.py --mode train \
     --phase finetune --checkpoint policy10.pt --max_iterations 100 \
     --num_envs 128 --threads 16 --warm_start_std 0 --run_name finetune_r3 2>&1 | tee finetune_r3.log
   ```

   - `--warm_start_std 0` keeps the learned action std. The default (0.5) raises it for exploration, which
     suits a new skill but disturbs a policy that already works.
   - `--run_name` only names the TensorBoard run under `logs/rsl_rl/liquid_transfer_v10/`.

3. Validate on seeds 17–21 before replacing `policy10.pt` (repeat for each seed).

   ```bash
   ../model_3_mujoco_ppo_known_positions/.venv/bin/python liquid_transfer_m3_v10.py --mode play \
     --checkpoint checkpoints/finetune.pt --num_envs 64 --max_steps 2700 --seed 17
   ```

   The output lists failures by stage and success by container offset. Compare against the table above.

4. If it is better, promote it: `cp checkpoints/finetune.pt policy10.pt`.

### Choosing what to train on

A phase is a mix of skills, defined in `PHASES` at the top of `liquid_transfer_m3_v10.py`. Each entry is
`((start_stage, end_stage), weight)`. To target a weakness, edit the `finetune` mix or add a new phase.

- Most remaining failures (39 of 53) are approach stalls, so the next round should raise the `(0, 0)`
  weight, for example `[((0, 5), 0.4), ((0, 0), 0.4), ((4, 5), 0.2)]`.
- Keep every skill you care about in the mix. Round 1 dropped `(0, 0)` and approach failures rose from
  9 to 70.
- To extend the placement range (for example toward layout B), train with `--beaker_range` and
  `--dish_range` above 0.175 and validate with the same values. The tip workspace clip (x ≥ 0.15 m)
  still applies.
- Reward terms and penalties are in the same file; changing them usually needs more iterations than a
  mix change.

To retrain from scratch, run `bash train_policy10.sh`, or `START=<phase> bash train_policy10.sh` to resume
the curriculum from a phase.
