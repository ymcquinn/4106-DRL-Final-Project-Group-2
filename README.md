# 4106-DRL-Final-Project-Group-2

by: Yema, Natkeeran, and Neil

**One-paragraph summary.** This project trains a deep reinforcement learning agent to perform one arm's worth of a self-driving lab task: aspirating nutrient solution from a beaker and dispensing it into a petri dish on a table with randomized container positions. The contribution is diagnostic: we found and closed a reward-specification exploit in a prior checkpoint that inflates the apparent success rate from 0.2% to 87.4% by sliding the pipette through the beaker wall. The full experiment, evidence, and reproduction commands are in [Experiment: v4 vs v6](#experiment-v4-vs-v6--reward-specification-diagnostic).

---

## Where this class project fits into Yema's broader solution architecture

### Master Agent (Bayesian Optimization lives here)
* Talks to the human. Owns the hypothesis. Schedules batches. Decides when a trial is done.
* Optimizes experimental conditions: glucose, peptone, pH, Mg²⁺, inoculation ratio.

### Supervisor Agent
* Coordinates between arms. Allocates tasks. Resolves conflicts. Safety interrupts.

### Arm Agents (this project — DRL)
* Each arm is a DRL agent:
  * State: tip position, target position, fluid level
  * Action: joint commands
  * Reward: reaching target + delivering correct volume
* Multiple arms = multiple agents, trained independently.

### Data Layer (Kafka / TML / Ollama)
* Streams: humidity, temperature, pH, culture viability, arm telemetry. Retention: 24 hours.

### Physical Layer
* Robotic arms + reservoirs + petri dishes + sensors.

---

## This project's scope: 1 arm

* **Human role:** places petri dishes containing *Cordyceps militaris* anywhere on the table.
* **Robot role:** one arm that uses a pipette to aspirate nutrient solution from a stationary beaker and dispense the liquid into a randomly located petri dish.

---

## Results at a glance

| Checkpoint | Reward version | Metric measured | Value |
|---|---|---|---|
| `policy.pt` (v4-trained) | v4 | Fluid-delivery rate | 87.4% |
| `policy.pt` (v4-trained) | v6 | **Task-success rate** | **0.2%** |
| Scripted controller | v6 | Task-success rate | 100% (16/16) |

**Reading the table:** the v4-trained policy delivers fluid 87.4% of the time under v4, but when evaluated under v6 with integrity gating, the same checkpoint succeeds only 0.2% of the time. The gap is the reward-specification exploit described below. The scripted controller confirms the fix is fair — it never penetrates, and v6 does not penalize it.

---

## Reward versions

- **`v4`** — original reward, 6 terms (progress, success, reach, time, overfill, action_rate). Measured metric: **fluid-delivery rate**. Admits the wall-penetration exploit. See [Experiment: v4 vs v6](#experiment-v4-vs-v6--reward-specification-diagnostic).
- **`v6`** — v4 + contact-force penalty + geometric penetration detection + safe-transit shaping + success integrity gate. Measured metric: **task-success rate**. Warmup (`--warmup_frac`) softens termination during the first N% of training so exploration can find the lift behavior.

---

## Training the DRL agents

* **Isaac Lab + teacher-student approach**
  * Sidesteps exploration via GPU parallelism and privileged state.
* **CPU-based PPO/SAC**
  * **Yema's tests**
    * Hit the sparse-reward wall.
    * Provides the diagnostic testbed.
    * `reach_env.py` — final state: beaker at (0.19, 0.44, 0.82), success threshold 0.08 m, linear reward.
    * `train_reach.py` — SAC configuration (buffer_size=100k, batch_size=256).
    * Training logs: PPO @ 0.14 m, PPO @ 0.105 m, SAC @ 0.105 m.
  * **Natkeeran's tests (Isaac Lab + teacher-student)**
    * Reported fluid-delivery rate: 96%.
    * Diagnostic: the policy delivers fluid by sliding the pipette through the beaker wall at low altitude. Under the v6 integrity gate, task-success rate is 0.2% (see [Experiment: v4 vs v6](#experiment-v4-vs-v6--reward-specification-diagnostic)).
    * This is the finding, not a defect in Natkeeran's training pipeline. The pipeline works; the original reward function did not distinguish fluid delivery from task success. v6 closes the gap.

---

## Setup

**Requirements:** Python 3.12, MuJoCo 3.10+, PyTorch, and the `franka_emika_panda/assets/` mesh folder inside this directory.

**Intel Mac note.** PyTorch wheels for x86_64 macOS stop at `torch==2.2.2`. If you're on Intel (2020 MacBook Pro and earlier), install `torch==2.2.2` explicitly:

```bash
python -m pip install "torch==2.2.2" --index-url https://download.pytorch.org/whl/cpu
python -m pip install mujoco==3.12.0 numpy
```

**Viewer.** On macOS, MuJoCo's interactive viewer requires `mjpython`, not `python`:

```bash
mjpython liquid_transfer_m3_v6.py --mode probe --num_envs 2 --viz
```

**Headless (no viewer).** Use `cgl` on macOS, `egl` on Linux:

```bash
MUJOCO_GL=cgl python liquid_transfer_m3_v6.py --mode scripted --num_envs 16 --max_steps 900
```

**Assets.** If `franka_emika_panda/assets/` is missing, download it from the MuJoCo Menagerie (Apache 2.0): https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda/assets

---

## Experiment: v4 vs v6 — reward specification diagnostic

**Status:** complete. **Date:** 2026-10-23. **Author:** Yema Quinn.

**Purpose:** demonstrate that the original reward function (v4) admits an exploit — the policy can deliver fluid to the dish while sliding the pipette through the beaker wall at low altitude — and that the patched reward (v6) catches it without penalizing correct behavior.

### Definitions used in this section

- **Fluid delivery** — liquid physically moves from the beaker to the dish. This is what v4's reward measures.
- **Task success** — fluid is delivered *and* the pipette never penetrates the beaker wall *and* no contact force exceeds 5 N. This is what the task requires.
- **Penetration event** — the pipette tip crosses the beaker wall at an altitude below the rim (z < 0.12 m).

### Reproduction

**Prerequisite:** `policy.pt` (the v4-trained checkpoint) must be in the working directory. Both runs use the same checkpoint, same seed (7), same physics, same 16 envs.

**Run A — v4 reward, fluid-delivery baseline:**

```bash
mjpython liquid_transfer_m3_v6.py --mode play --num_envs 16 --max_steps 1800 \
    --seed 7 --checkpoint policy.pt --reward_version v4 --viz
```

**Run B — v6 reward, task-success diagnostic:**

```bash
mjpython liquid_transfer_m3_v6.py --mode play --num_envs 16 --max_steps 1800 \
    --seed 7 --checkpoint policy.pt --reward_version v6 --warmup_frac 0.0 --viz
```

**Run C — scripted controller under v6 (fairness check):**

```bash
mjpython liquid_transfer_m3_v6.py --mode scripted --num_envs 4 --max_steps 900 \
    --reward_version v6 --warmup_frac 0.0 --viz
```

### Evidence — Run A (v4 policy, v4 reward)

```text
[Policy] loaded policy.pt: in_dim=13, layers=3, norm=yes
[Play] reward_version=v4; 1800 steps x 16 envs: 151 eps,
       outcomes {'success': 132, 'time_out': 19},
       success 87.4%, mean ep len 177; 63.1s
[Play] dish 0-5cm: 9/11 = 81.8%
[Play] dish 5-10cm: 24/24 = 100.0%
[Play] dish 10-15cm: 32/37 = 86.5%
[Play] dish 15-20cm: 44/54 = 81.5%
[Play] dish 20+cm: 23/25 = 92.0%
[Play] beaker 0-5cm: 6/8 = 75.0%
[Play] beaker 5-10cm: 21/23 = 91.3%
[Play] beaker 10-15cm: 36/43 = 83.7%
[Play] beaker 15-20cm: 52/57 = 91.2%
[Play] beaker 20+cm: 17/20 = 85.0%
```

**Reading Run A:** the v4 policy achieves an 87.4% *fluid-delivery rate*. This is **not** a task-success rate. The policy reaches this number by dipping the pipette to z ≈ 0.10 m, aspirating, and sliding horizontally to the dish at the same altitude. The beaker rim is at z = 0.12 m, so the tip travels through the beaker wall on exit. The pipette geom has no collision mesh (`contype=0, conaffinity=0`), so physics permits it. The reward only checks fluid in the dish, so the exploit is rewarded identically to the intended behavior.

### Evidence — Run B (v4 policy, v6 reward, warmup off)

```text
[Policy] loaded policy.pt: in_dim=13, layers=3, norm=yes
[Play] reward_version=v6; 1800 steps x 16 envs: 435 eps,
       outcomes {'integrity_violation': 434, 'success': 1},
       success 0.2%, mean ep len 65; 63.0s
[Play] dish 0-5cm: 0/21 = 0.0%
[Play] dish 5-10cm: 0/71 = 0.0%
[Play] dish 10-15cm: 0/118 = 0.0%
[Play] dish 15-20cm: 0/140 = 0.0%
[Play] dish 20+cm: 1/85 = 1.2%
[Play] beaker 0-5cm: 0/18 = 0.0%
[Play] beaker 5-10cm: 0/68 = 0.0%
[Play] beaker 10-15cm: 0/139 = 0.0%
[Play] beaker 15-20cm: 1/135 = 0.7%
[Play] beaker 20+cm: 0/75 = 0.0%
```

**Reading Run B:** the same v4-trained checkpoint, evaluated under v6 with warmup off, terminates on penetration in **434 of 435 episodes**. Task success is **0.2% (1/435)**. Mean episode length collapses from 177 to 65 steps — the policy is caught almost immediately after aspiration begins. The 1 remaining success is a rare episode where the policy happened to lift high enough to clear the rim; it was not isolated.

**The v4 policy's task-success rate, measured correctly: 0.2%.**

### Evidence — Run C (scripted controller, v6 reward, warmup off)

```text
[Scripted] terminations: {'success': 16}; mean episode len: 208; 32.1s
```

**Reading Run C:** the scripted controller — which lifts to z = 0.25 m before transiting — achieves 16/16 task success under v6 with warmup off, and zero penetration events. Identical to the result under warmup on. **v6 does not punish correct behavior.** The gate fires only on trajectories that would have caused penetration, and the scripted controller never penetrates.

### Conclusion

The v4-trained policy did not learn the task. It learned an exploit enabled by an incomplete reward function. Under v6 — same checkpoint, same seed, same physics, only the reward/success definition changed — the exploit is caught in 434 of 435 episodes and task success drops from "87.4% fluid delivery" to "0.2% task success."

The scripted control confirms the fix is fair: correct behavior still succeeds 100% under v6, warmup on or off.

### Video evidence

> **Video status:** recorded, not yet uploaded. Will be renamed and pushed to
> `docs/videos/` this weekend. Files referenced below are the planned names.

TO BE UPLOADED:

| File | What it shows | Reward | Warmup | Key numbers |
|---|---|---|---|---|
| `v5_v4_policy_v4_reward_warmup_on.mov` | Baseline: v4-trained policy delivering fluid under v4 reward. This is the "87.4% success" number in context. | v4 | on | 151 eps, 132 fluid deliveries (87.4%), mean 177 steps |
| `v4_v6_reward_warmup_on_penetration_logged.mov` | Same v4 policy under v6 reward with warmup on: penetrations are counted but not terminated. The exploit is visible but episodes continue. | v6 | on | 76 eps, 0% success, 25 overfill, 51 timeout, mean 355 steps |
| `v7_v6_reward_warmup_off_penetration.mov` | **The diagnostic.** Same v4 policy under v6 reward with warmup off: penetration terminates immediately. 434 of 435 episodes end this way. | v6 | **off** | 435 eps, 0.2% task success, 434 integrity violations, mean 65 steps |
| `v8_scripted_v6_warmup_off_success.mov` | **The fairness check.** Scripted controller under v6 with warmup off. Lifts to z = 0.25 m before transiting. 16/16 task success, zero penetration. | v6 | **off** | 16 eps, 100% task success, mean 208 steps |

**On reproducibility:** the v4-policy-under-v4-reward run was recorded twice — once before and once after the rsl_rl API bypass patch (`run_play_manual`). Output was identical (87.4%, same per-bin breakdown). The bypass changed code structure, not physics or reward. Similarly for the v6-warmup-on run (both recorded, identical). Only the archived copies differ.

**Archived in `docs/videos/archive/`:** earlier probe and false-start recordings, kept for working history.

### Reproducibility notes

- **Same checkpoint, same seed, same physics.** The only difference between Run A and Run B is `--reward_version`. Warmup is off in both (`--warmup_frac` only affects v6, and Run A doesn't use it).
- **Penetration detection is geometric, not contact-based.** It tracks the tip site's XY distance from the beaker centre and its altitude each step. It does not rely on collision geometry, which is why it catches the exploit even though the pipette has no collision mesh.
- **The v6 success criterion is fluid delivery AND no penetration AND contact force below threshold.** Any of those three failing means task success does not fire.

---

## Contributions

- **Yema Quinn** — architecture, reward design (v4 → v6), environment (MuJoCo, Gymnasium), diagnostic experiment, report, team lead, environmental controls lead.
- **Natkeeran** — Isaac Lab + teacher-student training pipeline, original checkpoint (`policy.pt`), baseline results. The v4 checkpoint was the starting point for the v6 diagnostic.
- **Neil** — environmental-disturbance curriculum proposal (moving containers as a difficulty knob), directly testable under v6.

Course: 4106 Deep Reinforcement Learning, University of Toronto School of Continuing Studies, Fall 2026. Instructor: Dr. Sebastian Maurice.

---

## Still pending

- **v6 training from scratch.** The v6 reward has been validated end-to-end: scripted control succeeds 100%, the v4 checkpoint is caught 434/435 times. What has not been shown: that a PPO agent can *learn* the safe task under v6. Sparse-reward exploration is still the bottleneck at 3-DOF scale on CPU. Two candidate paths under evaluation: (a) GPU-parallel training under Isaac Lab, (b) curriculum with a reduced-DOF arm as a warm-up phase.
- **Contact-force history plot.** Logged in v6 metrics but not yet visualized.
- **Physical-fix ablation.** Add a collision sphere to the pipette tip in `panda_pipette.xml` and re-run the v4 checkpoint under v4 reward. Demonstrates that the exploit is a physics-vs-reward choice, not a rendering artifact.
