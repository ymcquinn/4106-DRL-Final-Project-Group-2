# 4106-DRL-Final-Project-Group-2

by: Yema, Natkeeran, and Neil

# Where this class project fit's into Yema's broader Solution Architecture

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

# This Project's Scope: 1 Arm
* Human role: places petri dishes containing Cordyceps militaris anywhere on the table.
* Robot role: one arm that uses a pipette to aspirate nutrient solution from a stationary beaker and dispense the liquid into a randomly located petri dish 

## Training the DRL Agents
* Isaac Lab + teacher-student approach 
    * sidesteps exploration via GPU parallelism and privileged state
* CPU-based PPO/SAC
  	* Yema's tests 
    	* hit the sparse-reward wall mine 
   	 	* provides the diagnostic testbed
   		* reach_env.py — final state: beaker at (0.19, 0.44, 0.82), success threshold 0.08m, linear reward
   	 	* train_reach.py — SAC configuration (buffer_size=100k, batch_size=256)
   		* Training logs: PPO @ 0.14m, PPO @ 0.105m, SAC @ 0.105m
 	* Natkeeran's tests
		* The agent realized that the fastest way to get liquid from the beaker to the Petri dish is to smash through the beaker lol, making the 96% success rate actually a 0% success rate
 
## Experiment: v4 vs v6 — reward specification diagnostic

**Status:** complete. Date: 2026-10-23. Author: Yema Quinn.

**Purpose:** demonstrate that the original reward function (v4) admits an exploit — the policy can deliver fluid to the dish while sliding the pipette through the beaker wall at low altitude — and that the patched reward (v6) catches it without penalizing correct behavior.

### Definitions used in this section:

**Fluid delivery** — liquid physically moves from the beaker to the dish. This is what v4's reward measures.
**Task success** — fluid is delivered and the pipette never penetrates the beaker wall and no contact force exceeds 5 N. This is what the task requires.
**Penetration event** — the pipette tip crosses the beaker wall at an altitude below the rim (z < 0.12 m).

### Reproduction

**Prerequisite:** policy.pt (the v4-trained checkpoint) must be in the working directory. Both runs use the same checkpoint, same seed (7), same physics, same 16 envs.

**Run A — v4 reward, fluid-delivery baseline:**

bash
`mjpython liquid_transfer_m3_v6.py --mode play --num_envs 16 --max_steps 1800 \
    --seed 7 --checkpoint policy.pt --reward_version v4 --viz`

**Run B — v6 reward, task-success diagnostic:**

bash
`mjpython liquid_transfer_m3_v6.py --mode play --num_envs 16 --max_steps 1800 \
    --seed 7 --checkpoint policy.pt --reward_version v6 --warmup_frac 0.0 --viz`

**Run C — scripted controller under v6 (fairness check):**

bash
`mjpython liquid_transfer_m3_v6.py --mode scripted --num_envs 4 --max_steps 900 \
    --reward_version v6 --warmup_frac 0.0 --viz`

### Evidence — Run A (v4 policy, v4 reward)

text
`[Policy] loaded policy.pt: in_dim=13, layers=3, norm=yes
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
[Play] beaker 20+cm: 17/20 = 85.0%`

**Reading Run A:** the v4 policy achieves an 87.4% *fluid-delivery rate*. This is not a task-success rate. The policy reaches this number by dipping the pipette to z ≈ 0.10 m, aspirating, and sliding horizontally to the dish at the same altitude. The beaker rim is at z = 0.12 m, so the tip travels through the beaker wall on exit. The pipette geom has no collision mesh `(contype=0, conaffinity=0)`, so physics permits it. The reward only checks fluid in the dish, so the exploit is rewarded identically to the intended behavior.

### Evidence — Run B (v4 policy, v6 reward, warmup off)

text

`[Policy] loaded policy.pt: in_dim=13, layers=3, norm=yes
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
[Play] beaker 20+cm: 0/75 = 0.0%`

**Reading Run B:** the same v4-trained checkpoint, evaluated under v6 with warmup off, terminates on penetration in **434 of 435 episodes**. Task success is **0.2% (1/435)**. Mean episode length collapses from 177 to 65 steps — the policy is caught almost immediately after aspiration begins. The 1 remaining "success" is a rare episode where the policy happened to lift high enough to clear the rim; it was not isolated.

**The v4 policy's task-success rate, measured correctly: 0.2%.**

### Evidence — Run C (scripted controller, v6 reward, warmup off)

text
`[Scripted] terminations: {'success': 16}; mean episode len: 208; 32.1s`

**Reading Run C:** the scripted controller — which lifts to z = 0.25 m before transiting — achieves 16/16 task success under v6 with warmup off, and zero penetration events. Identical to the result under warmup on. v6 does not punish correct behavior. The gate fires only on trajectories that would have caused penetration, and the scripted controller never penetrates.

### Conclusion

The v4-trained policy did not learn the task. It learned an exploit enabled by an incomplete reward function. Under v6 — same checkpoint, same seed, same physics, only the reward/success definition changed — the exploit is caught in 434 of 435 episodes and task success drops from "87.4% fluid delivery" to "0.2% task success."

The scripted control confirms the fix is fair: correct behavior still succeeds 100% under v6, warmup on or off.

Video evidence

Screen recordings of Runs B and C are in `docs/videos/`:

`exploit_penetration_v6.mp4` — v4 policy triggering penetration terminations under v6 (warmup off). The pipette is visible passing through the beaker wall.
`scripted_success_v6.mp4` — scripted controller executing correctly under v6 (warmup off). No penetration.

### Reproducibility notes

* **Same checkpoint, same seed, same physics.** The only difference between Run A and Run B is `--reward_version`. Warmup is off in both (`--warmup_frac` only affects v6, and Run A doesn't use it).
* **Penetration detection is geometric, not contact-based.** It tracks the tip site's XY distance from the beaker centre and its altitude each step. It does not rely on collision geometry, which is why it catches the exploit even though the pipette has no collision mesh.
* **The v6 success criterion is fluid delivery AND no penetration AND contact force below threshold.** Any of those three failing means task success does not fire.
What else to add to README (short)


## Reward versions

- `v4` — original reward, 6 terms (progress, success, reach, time, overfill, action_rate).
  Measured metric: **fluid-delivery rate**. Admits the wall-penetration exploit.
  See "Experiment: v4 vs v6" below.
- `v6` — v4 + contact-force penalty + geometric penetration detection + safe-transit
  shaping + success integrity gate. Measured metric: **task-success rate**.
  Warmup (`--warmup_frac`) softens termination during the first N% of training so
  exploration can find the lift behavior.
