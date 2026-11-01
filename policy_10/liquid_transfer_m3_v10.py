#liquid_transfer_m3_v10.py

"""
Policy 10 (MuJoCo, model 3 scene): liquid transfer learned with a staged curriculum, with the
learned policy in control of the robot at every step.

Changes from Trained and Validated/liquid_transfer_m3_v6_trained.py:
  * No scripted guidance in env.step(). The v6 end_to_end block overwrote every policy action
    with a proportional controller; here the clipped policy action is the action applied.
  * The Stage 0->1 XY target snap is removed (it moved the arm setpoint on the policy's behalf).
  * One observation layout for every curriculum phase, so each phase warm-starts from the
    previous phase's checkpoint (v6 used 16 dims for isolated curricula and 19 for end_to_end).
  * Curriculum skills are slices of the same six-stage task: (start stage, end stage). Start
    states are settled at randomized container positions (v6 isolated curricula used one fixed
    joint pose, so they only worked with the beaker at its nominal position). Later phases replay
    earlier skills to avoid forgetting.
  * Potential-based progress shaping toward the active stage waypoint. The v6 safety model is kept:
    contact-force penalty, penetration detection, integrity-gated success, unsafe termination.

Stages: 0 approach above beaker, 1 descend + aspirate, 2 lift out of beaker,
        3 transit to above dish, 4 descend + dispense, 5 lift out of dish.

Run from this folder (uses the venv created by ../model_3_mujoco_ppo_known_positions/setup_env.sh):
    PY=../model_3_mujoco_ppo_known_positions/.venv/bin/python
    Scripted baseline: $PY liquid_transfer_m3_v10.py --mode scripted --num_envs 16 --max_steps 900
    Train curriculum:  bash train_policy10.sh
    One phase:         $PY liquid_transfer_m3_v10.py --mode train --phase approach --max_iterations 150
    Eval:              $PY liquid_transfer_m3_v10.py --mode play --checkpoint policy10.pt \
                           --num_envs 64 --max_steps 2700 --seed 17
    Ablation:          add --action_override zeros to the eval command (policy output discarded)
"""

import argparse
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Curriculum phases: each entry is ((start_stage, end_stage), sampling weight).
# An episode starts in start_stage and succeeds when end_stage is completed safely.
PHASES = {
    "approach":      [((0, 0), 1.0)],
    "aspirate_lift": [((1, 2), 0.7), ((0, 0), 0.3)],
    "transit":       [((3, 3), 0.6), ((1, 2), 0.2), ((0, 0), 0.2)],
    "dispense_lift": [((4, 5), 0.55), ((3, 3), 0.15), ((1, 2), 0.15), ((0, 0), 0.15)],
    "chain":         [((0, 5), 0.4), ((1, 5), 0.15), ((2, 5), 0.15), ((3, 5), 0.15), ((4, 5), 0.15)],
    "end_to_end":    [((0, 5), 0.8), ((1, 5), 0.05), ((2, 5), 0.05), ((3, 5), 0.05), ((4, 5), 0.05)],
    "eval":          [((0, 5), 1.0)],
}

parser = argparse.ArgumentParser(description="Liquid transfer, Policy 10 (curriculum PPO, policy in control).")
parser.add_argument("--mode", choices=["scripted", "train", "play"], default="play")
parser.add_argument("--phase", choices=list(PHASES), default=None,
                    help="Curriculum phase. Required for train; play and scripted default to 'eval'.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_steps", type=int, default=1800)
parser.add_argument("--max_iterations", type=int, default=200)
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--episode_length_s", type=float, default=20.0)
parser.add_argument("--dish_range", type=float, default=0.175)
parser.add_argument("--beaker_range", type=float, default=0.175)
parser.add_argument("--beaker_offset", type=str, default="",
                    help="Fixed beaker XY offset 'dx,dy' in m from nominal (0.40,-0.20); overrides --beaker_range.")
parser.add_argument("--dish_offset", type=str, default="",
                    help="Fixed dish XY offset 'dx,dy' in m from nominal (0.40,0.20); overrides --dish_range.")
parser.add_argument("--warmup_frac", type=float, default=0.0,
                    help="Fraction of this run during which unsafe events are penalized but do not terminate.")
parser.add_argument("--warm_start_std", type=float, default=0.5,
                    help="train with --checkpoint: raise the action std to at least this value.")
parser.add_argument("--action_override", choices=["none", "zeros", "random"], default="none",
                    help="play only: replace the policy output (ablation check).")
parser.add_argument("--trace", action="store_true", help="play only: print env 0 actions at stage changes.")
parser.add_argument("--viz", action="store_true")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--threads", type=int, default=0)
# Let "--dish_offset -0.05,0.1" work: argparse would read the negative value as a flag.
_argv = sys.argv[1:]
for _i in range(len(_argv) - 1):
    if _argv[_i] in ("--beaker_offset", "--dish_offset") and _argv[_i + 1][:1] == "-":
        _argv[_i], _argv[_i + 1] = f"{_argv[_i]}={_argv[_i + 1]}", ""
args_cli = parser.parse_args([a for a in _argv if a != ""])
if args_cli.phase is None:
    if args_cli.mode == "train":
        parser.error("--mode train needs --phase")
    args_cli.phase = "eval"

if not args_cli.viz:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(HERE, "liquid_transfer_scene.xml")

# Geometry (same scene as v6_trained: hollow beaker walls + pipette tip collision sphere)
BEAKER_POS = (0.4, -0.2, 0.06)
BEAKER_RADIUS = 0.05
BEAKER_RIM_Z = 0.12
DISH_POS = (0.4, 0.2, 0.01)
DISH_RADIUS = 0.08
MIN_REACH_M = 0.29
MIN_SEPARATION_M = 0.18
DISTANCE_BINS_CM = (5, 10, 15, 20)
TIP_DOWN_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])

# Liquid
BEAKER_INITIAL_ML = 10.0
PIPETTE_CAPACITY_ML = 10.0
DISH_CAPACITY_ML = 10.0
FLOW_ML_PER_STEP = 0.2
TARGET_ML = 2.0
SUCCESS_TOL_ML = 0.3
ASPIRATE_MAX_TIP_Z = 0.11
DISPENSE_MAX_TIP_Z = 0.10

# Control
CONTROL_HZ = 30.0
DECIMATION = 4
PHYSICS_SUBSTEPS = 2
STEP_SIZE = 0.01
TARGET_LEASH_M = 0.03                 # setpoint may lead the tip by at most 3 cm per axis
HOME_TIP_POS = (0.28, 0.0, 0.38)
WORKSPACE_MIN = (0.15, -0.45, 0.03)
WORKSPACE_MAX = (0.65, 0.45, 0.45)
IK_LAMBDA = 0.01
HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
RESET_JOINT_NOISE = 0.1
SETTLE_STEPS = 80
OBS_POS_CENTER = np.array([0.4, 0.0, 0.2])
OBS_POS_SCALE = 0.2

# Stage waypoints (same heights and tolerances as v6 end_to_end)
CLEAR_Z = BEAKER_RIM_Z + 0.08         # 0.20 m: above beaker, transit height, above dish
ASPIRATE_GOAL_Z = 0.08
DISPENSE_GOAL_Z = 0.06
XY_TOL_M = 0.020
Z_TOL_M = 0.020
DESCENT_MAX_ACTION = 0.35
NUM_STAGES = 6

# Safety (v6)
CONTACT_FORCE_THRESHOLD_N = 5.0
CONTACT_FORCE_CAP_N = 100.0

# Reward weights
W_PROGRESS_PER_M = 60.0               # potential-based: decrease in distance to the stage waypoint
W_STAGE = 10.0                        # each stage completed
W_FLUID_PER_ML = 5.0                  # liquid aspirated in stage 1 / dispensed in stage 4
W_SUCCESS = 100.0
W_TIME = -0.02
W_CENTERLINE_PER_M = -10.0            # XY offset from the container axis while low inside it
W_LOW_TRAVEL_PER_M = -20.0            # below clearance height during horizontal stages 0 and 3
W_FAST_DESCENT = -2.0
W_ACTION = -0.01
W_ACTION_RATE = -0.02
W_CONTACT_SOFT_PER_N = -0.05          # any container contact (capped at 20 N)
W_CONTACT_PER_N = -0.5                # v6: contact force above 5 N
W_PENETRATION_DENSE = -5.0            # v6
W_UNSAFE_TERMINATE = -50.0            # v6

REWARD_TERMS = ("progress", "stage", "fluid", "success", "time", "centerline", "low_travel", "fast_descent",
                "action", "contact", "penetration")
TERMS = ("time_out", "success", "tip_below_table", "integrity_violation", "excessive_contact_force")


def quat_mul(a, b):
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2], axis=1)

def quat_conj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])

def axis_angle_from_quat(q):
    q = np.where(q[:, :1] < 0, -q, q)
    v = q[:, 1:]
    s = np.linalg.norm(v, axis=1)
    angle = 2.0 * np.arctan2(s, q[:, 0])
    scale = np.where(s > 1e-8, angle / np.maximum(s, 1e-12), 2.0)
    return v * scale[:, None]

def distance_bin(d):
    return np.searchsorted(np.array(DISTANCE_BINS_CM) / 100.0, d, side="right")

def distance_bin_label(b):
    lo = 0 if b == 0 else DISTANCE_BINS_CM[b - 1]
    return f"{lo}+cm" if b >= len(DISTANCE_BINS_CM) else f"{lo}-{DISTANCE_BINS_CM[b]}cm"


class LiquidTransferV10Env:
    def __init__(self, num_envs, phase="eval", seed=42, episode_length_s=20.0, dish_range=0.175,
                 beaker_range=0.175, fixed_beaker_offset=None, fixed_dish_offset=None,
                 device="cpu", threads=0, auto_reset=True, warmup_steps=0):
        self.num_envs = num_envs
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.dish_range, self.beaker_range = dish_range, beaker_range
        self.fixed_beaker_offset, self.fixed_dish_offset = fixed_beaker_offset, fixed_dish_offset
        self.auto_reset = auto_reset
        self.step_dt = 1.0 / CONTROL_HZ
        self.episode_length_s = episode_length_s
        self.max_episode_length = math.ceil(episode_length_s / self.step_dt)
        self.skills = np.array([s for s, _ in PHASES[phase]], dtype=np.int64)
        w = np.array([w for _, w in PHASES[phase]], dtype=float)
        self.skill_probs = w / w.sum()
        self.warmup_steps = warmup_steps
        self.total_env_steps = 0

        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        assert abs(self.model.opt.timestep * PHYSICS_SUBSTEPS * DECIMATION - self.step_dt) < 1e-6
        self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)]
        m = self.model
        self.hand_id = m.body("hand").id
        self.tip_site = m.site("pipette_tip").id
        self.container_geom_ids = {m.geom("beaker_geom").id, m.geom("dish_geom").id,
                                   *{m.geom(f"beaker_wall_{i:02d}").id for i in range(16)}}
        self.beaker_mocap = m.body_mocapid[m.body("beaker").id]
        self.dish_mocap = m.body_mocapid[m.body("petri_dish").id]
        self.joint_range = m.jnt_range.copy()
        assert np.allclose(m.key_qpos[m.key("home").id], HOME_QPOS)

        n = num_envs
        self.target_pos = np.tile(np.array(HOME_TIP_POS), (n, 1))
        self.ws_min, self.ws_max = np.array(WORKSPACE_MIN), np.array(WORKSPACE_MAX)
        self.action = np.zeros((n, 3)); self.prev_action = np.zeros((n, 3))
        self._jacp = np.zeros((3, m.nv)); self._jacr = np.zeros((3, m.nv))
        self.target_ml = np.full(n, TARGET_ML); self.pipette_ml = np.zeros(n)
        self.dish_ml = np.zeros(n); self.beaker_ml = np.full(n, BEAKER_INITIAL_ML)
        self.prev_pipette_ml = np.zeros(n); self.prev_dish_ml = np.zeros(n)
        self.tip_pos = np.zeros((n, 3)); self.prev_tip_pos = np.zeros((n, 3))
        self.beaker_pos = np.zeros((n, 3)); self.dish_pos = np.zeros((n, 3))
        self.dish_offset = np.zeros((n, 2)); self.beaker_offset = np.zeros((n, 2))
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)

        self.task_stage = np.zeros(n, dtype=np.int64)
        self.start_stage = np.zeros(n, dtype=np.int64)
        self.end_stage = np.full(n, NUM_STAGES - 1, dtype=np.int64)
        self.max_stage_this_ep = np.zeros(n, dtype=np.int64)
        self.stage_advanced_this_step = np.zeros(n, dtype=bool)
        self.step_progress = np.zeros(n)

        self.prev_dist_beaker = np.zeros(n); self.prev_tip_z = np.zeros(n)
        self.contact_force_this_step = np.zeros(n)
        self.penetration_event_this_step = np.zeros(n, dtype=bool)
        self.penetration_count_this_ep = np.zeros(n, dtype=int)
        self.max_contact_force_this_ep = np.zeros(n)

        self.terms_done = {k: np.zeros(n, dtype=bool) for k in TERMS}
        self.terminated = np.zeros(n, dtype=bool); self.time_outs = np.zeros(n, dtype=bool)
        self.reward_sums = {k: np.zeros(n) for k in REWARD_TERMS}
        self.extras = {}; self.obs_buf = {}
        self.pool = ThreadPoolExecutor(max_workers=threads or min(8, os.cpu_count() or 1))
        self.viewer = None
        self.reset()

    # ---------------------------------------------------------------- physics / control
    def _ik_tick(self):
        n = self.num_envs
        jac = np.zeros((n, 6, 7)); err = np.zeros((n, 6)); cur_quat = np.zeros((n, 4))
        for i, d in enumerate(self.datas):
            mujoco.mj_jacSite(self.model, d, self._jacp, self._jacr, self.tip_site)
            jac[i, :3] = self._jacp[:, :7]; jac[i, 3:] = self._jacr[:, :7]
            err[i, :3] = self.target_pos[i] - d.site_xpos[self.tip_site]
            cur_quat[i] = d.xquat[self.hand_id]
        q_err = quat_mul(np.tile(TIP_DOWN_QUAT_WXYZ, (n, 1)), quat_conj(cur_quat))
        err[:, 3:] = axis_angle_from_quat(q_err)
        jjt = jac @ jac.transpose(0, 2, 1) + (IK_LAMBDA**2) * np.eye(6)
        dq = jac.transpose(0, 2, 1) @ np.linalg.solve(jjt, err[:, :, None])
        for i, d in enumerate(self.datas):
            d.ctrl[:7] = d.qpos[:7] + dq[i, :, 0]
            d.ctrl[7] = 255.0

    def _settle_tip_target(self, idx, goal, steps=SETTLE_STEPS):
        """Move one reset state to a Cartesian tip goal before exposing it to PPO."""
        d = self.datas[idx]
        jacp = np.zeros((3, self.model.nv)); jacr = np.zeros((3, self.model.nv))
        for _ in range(steps):
            mujoco.mj_jacSite(self.model, d, jacp, jacr, self.tip_site)
            jac = np.zeros((6, 7)); jac[:3] = jacp[:, :7]; jac[3:] = jacr[:, :7]
            err = np.zeros(6)
            err[:3] = goal - d.site_xpos[self.tip_site]
            q_err = quat_mul(TIP_DOWN_QUAT_WXYZ[None, :], quat_conj(d.xquat[self.hand_id][None, :]))
            err[3:] = axis_angle_from_quat(q_err)[0]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + (IK_LAMBDA ** 2) * np.eye(6), err)
            d.ctrl[:7] = d.qpos[:7] + dq
            d.ctrl[7] = 255.0
            for _ in range(PHYSICS_SUBSTEPS * DECIMATION):
                mujoco.mj_step(self.model, d)
        mujoco.mj_forward(self.model, d)

    def _step_physics(self, idx):
        d = self.datas[idx]
        for _ in range(PHYSICS_SUBSTEPS):
            mujoco.mj_step(self.model, d)

    def _apply_actions(self):
        for _ in range(DECIMATION):
            self._ik_tick()
            list(self.pool.map(self._step_physics, range(self.num_envs)))

    def _refresh_poses(self, ids=None):
        for i in (range(self.num_envs) if ids is None else ids):
            d = self.datas[i]
            self.tip_pos[i] = d.site_xpos[self.tip_site]
            self.beaker_pos[i] = d.mocap_pos[self.beaker_mocap]
            self.dish_pos[i] = d.mocap_pos[self.dish_mocap]

    # ---------------------------------------------------------------- safety (v6)
    def _update_contact_forces(self):
        cf = np.zeros(self.num_envs)
        buf = np.zeros(6)
        for i, d in enumerate(self.datas):
            total = 0.0
            for c in range(d.ncon):
                con = d.contact[c]
                if con.geom1 in self.container_geom_ids or con.geom2 in self.container_geom_ids:
                    mujoco.mj_contactForce(self.model, d, c, buf)
                    total += abs(buf[0])
            cf[i] = min(total, CONTACT_FORCE_CAP_N)
        self.contact_force_this_step = cf
        self.max_contact_force_this_ep = np.maximum(self.max_contact_force_this_ep, cf)

    def _update_penetration(self):
        d_b = np.linalg.norm(self.tip_pos[:, :2] - self.beaker_pos[:, :2], axis=1)
        z = self.tip_pos[:, 2]
        prev_in_low = (self.prev_dist_beaker < BEAKER_RADIUS) & (self.prev_tip_z < BEAKER_RIM_Z)
        leave_low = prev_in_low & (d_b >= BEAKER_RADIUS) & (z < BEAKER_RIM_Z)
        self.penetration_event_this_step = leave_low
        self.penetration_count_this_ep += leave_low.astype(int)
        self.prev_dist_beaker = d_b
        self.prev_tip_z = z

    def _integrity_ok(self):
        return (self.penetration_count_this_ep == 0) & (self.max_contact_force_this_ep < CONTACT_FORCE_THRESHOLD_N)

    # ---------------------------------------------------------------- liquid
    def _update_fluid(self):
        tip = self.tip_pos
        in_beaker = ((np.linalg.norm(tip[:, :2] - self.beaker_pos[:, :2], axis=1) < BEAKER_RADIUS - 0.005)
                     & (tip[:, 2] < ASPIRATE_MAX_TIP_Z))
        # Aspiration stops at the requested volume.
        room = np.minimum(PIPETTE_CAPACITY_ML - self.pipette_ml, np.maximum(0.0, self.target_ml - self.pipette_ml))
        take = np.minimum(np.where(in_beaker, FLOW_ML_PER_STEP, 0.0), np.minimum(self.beaker_ml, room))
        self.beaker_ml -= take; self.pipette_ml += take
        over_dish = ((np.linalg.norm(tip[:, :2] - self.dish_pos[:, :2], axis=1) < DISH_RADIUS - 0.01)
                     & (tip[:, 2] < DISPENSE_MAX_TIP_Z))
        give = np.minimum(np.where(over_dish, FLOW_ML_PER_STEP, 0.0),
                          np.minimum(self.pipette_ml, DISH_CAPACITY_ML - self.dish_ml))
        self.pipette_ml -= give; self.dish_ml += give

    # ---------------------------------------------------------------- six-stage task
    def _stage_goal(self, stage):
        b_high = self.beaker_pos.copy(); b_high[:, 2] = CLEAR_Z
        b_low = self.beaker_pos.copy(); b_low[:, 2] = ASPIRATE_GOAL_Z
        d_high = self.dish_pos.copy(); d_high[:, 2] = CLEAR_Z
        d_low = self.dish_pos.copy(); d_low[:, 2] = DISPENSE_GOAL_Z
        choices = np.stack([b_high, b_low, b_high, d_high, d_low, d_high], axis=1)
        return choices[np.arange(self.num_envs), stage]

    def _advance_stage(self):
        """Same transition conditions as v6 end_to_end, one stage per step, no setpoint snap."""
        bxy = np.linalg.norm(self.tip_pos[:, :2] - self.beaker_pos[:, :2], axis=1)
        dxy = np.linalg.norm(self.tip_pos[:, :2] - self.dish_pos[:, :2], axis=1)
        z = self.tip_pos[:, 2]
        conditions = (
            (bxy <= XY_TOL_M) & (z >= CLEAR_Z - Z_TOL_M),                     # 0 -> 1 centered above beaker
            self.pipette_ml >= self.target_ml - SUCCESS_TOL_ML,                  # 1 -> 2 aspirated
            (bxy <= XY_TOL_M) & (z >= CLEAR_Z - Z_TOL_M),                     # 2 -> 3 lifted clear
            (dxy <= XY_TOL_M) & (np.abs(z - CLEAR_Z) <= Z_TOL_M),             # 3 -> 4 centered above dish
            self.dish_ml >= self.target_ml - SUCCESS_TOL_ML,                     # 4 -> 5 dispensed
            np.zeros(self.num_envs, dtype=bool),                                 # 5 ends via success
        )
        cond = np.stack(conditions, axis=1)[np.arange(self.num_envs), self.task_stage]
        self.stage_advanced_this_step = cond
        self.task_stage = self.task_stage + cond
        self.max_stage_this_ep = np.maximum(self.max_stage_this_ep, self.task_stage)

    def _task_success(self):
        dxy = np.linalg.norm(self.tip_pos[:, :2] - self.dish_pos[:, :2], axis=1)
        full = ((self.task_stage == 5) & (dxy <= XY_TOL_M) & (self.tip_pos[:, 2] >= CLEAR_Z - Z_TOL_M)
                & (np.abs(self.dish_ml - self.target_ml) < SUCCESS_TOL_ML) & (self.pipette_ml <= SUCCESS_TOL_ML))
        partial = self.task_stage > self.end_stage
        return np.where(self.end_stage == NUM_STAGES - 1, full, partial) & self._integrity_ok()

    def _in_warmup(self):
        return self.total_env_steps < self.warmup_steps

    def _compute_terminations(self):
        self.terms_done["time_out"] = self.episode_length_buf.numpy() >= self.max_episode_length
        self.terms_done["success"] = self._task_success()
        self.terms_done["tip_below_table"] = self.tip_pos[:, 2] < 0.0
        if self._in_warmup():
            self.terms_done["integrity_violation"][:] = False
            self.terms_done["excessive_contact_force"][:] = False
        else:
            self.terms_done["integrity_violation"] = self.penetration_event_this_step.copy()
            self.terms_done["excessive_contact_force"] = self.max_contact_force_this_ep >= CONTACT_FORCE_THRESHOLD_N
        skill_over = self.task_stage > self.end_stage
        self.terminated = (self.terms_done["success"] | self.terms_done["tip_below_table"]
                           | self.terms_done["integrity_violation"] | self.terms_done["excessive_contact_force"]
                           | skill_over)
        self.time_outs = self.terms_done["time_out"] & ~self.terminated

    def _compute_rewards(self, stage):
        """stage: the stage that was active while this step's action was executed."""
        n = self.num_envs
        z = self.tip_pos[:, 2]
        bxy = np.linalg.norm(self.tip_pos[:, :2] - self.beaker_pos[:, :2], axis=1)
        dxy = np.linalg.norm(self.tip_pos[:, :2] - self.dish_pos[:, :2], axis=1)
        aspirated = np.where(stage == 1, np.maximum(0.0, self.pipette_ml - self.prev_pipette_ml), 0.0)
        dispensed = np.where(stage == 4, np.maximum(0.0, self.dish_ml - self.prev_dish_ml), 0.0)
        in_beaker_low = np.isin(stage, (1, 2)) & (z < BEAKER_RIM_Z + 0.02)
        in_dish_low = np.isin(stage, (4, 5)) & (z < DISPENSE_MAX_TIP_Z + 0.02)
        off_axis = (np.where(in_beaker_low, np.maximum(0.0, bxy - 0.01), 0.0)
                    + np.where(in_dish_low, np.maximum(0.0, dxy - 0.01), 0.0))
        descent_excess = np.maximum(0.0, -self.action[:, 2] - DESCENT_MAX_ACTION)
        cf = self.contact_force_this_step
        unsafe = self.terms_done["integrity_violation"] | self.terms_done["excessive_contact_force"]
        r = {
            "progress": W_PROGRESS_PER_M * self.step_progress,
            "stage": W_STAGE * self.stage_advanced_this_step,
            "fluid": W_FLUID_PER_ML * (aspirated + dispensed),
            "success": W_SUCCESS * self.terms_done["success"],
            "time": np.full(n, W_TIME),
            "centerline": W_CENTERLINE_PER_M * off_axis,
            # v6 forced transit to stay at clearance height; here the policy is penalized instead.
            "low_travel": np.where(np.isin(stage, (0, 3)),
                                   W_LOW_TRAVEL_PER_M * np.maximum(0.0, CLEAR_Z - Z_TOL_M - z), 0.0),
            "fast_descent": np.where(np.isin(stage, (1, 4)), W_FAST_DESCENT * descent_excess ** 2, 0.0),
            "action": (W_ACTION * np.sum(self.action ** 2, axis=1)
                       + W_ACTION_RATE * np.sum((self.action - self.prev_action) ** 2, axis=1)),
            "contact": (W_CONTACT_SOFT_PER_N * np.minimum(cf, 20.0)
                        + W_CONTACT_PER_N * np.maximum(0.0, cf - CONTACT_FORCE_THRESHOLD_N)),
            "penetration": (W_PENETRATION_DENSE * self.penetration_event_this_step
                            + W_UNSAFE_TERMINATE * unsafe),
        }
        total = np.zeros(n)
        for k, v in r.items():
            self.reward_sums[k] += v
            total += v
        return total

    # ---------------------------------------------------------------- observations
    def _compute_obs(self):
        # Fixed scaling to roughly unit range. A running normalizer is unsafe with this curriculum:
        # inputs that are constant in early phases (stage one-hot, liquid volumes) get std ~ 0 and
        # blow up when a later phase first activates them.
        f32 = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=self.device)
        beaker_ref = self.beaker_pos.copy(); beaker_ref[:, 2] = BEAKER_RIM_Z
        dish_ref = self.dish_pos.copy(); dish_ref[:, 2] = DISH_POS[2]
        fluid = np.stack([self.target_ml, self.pipette_ml, self.dish_ml], axis=1) / TARGET_ML
        stage_obs = np.eye(NUM_STAGES, dtype=np.float32)[self.task_stage]
        policy = np.concatenate([(self.tip_pos - OBS_POS_CENTER) / OBS_POS_SCALE,
                                 (self.tip_pos - beaker_ref) / OBS_POS_SCALE,
                                 (self.tip_pos - dish_ref) / OBS_POS_SCALE,
                                 (self.target_pos - self.tip_pos) / TARGET_LEASH_M,
                                 fluid, self.action, stage_obs], axis=1)
        joint_vel = np.stack([d.qvel[:9] for d in self.datas])
        critic = np.concatenate([policy, joint_vel, self.contact_force_this_step[:, None] / 20.0], axis=1)
        self.obs_buf = {"policy": f32(policy), "critic": f32(critic)}
        return self.obs_buf

    # ---------------------------------------------------------------- resets
    def _sample_offsets(self, k):
        b_nom, d_nom = np.array(BEAKER_POS[:2]), np.array(DISH_POS[:2])
        fixed = lambda off: (lambda m: np.tile(np.array(off, dtype=float), (m, 1)))
        draw_b = (fixed(self.fixed_beaker_offset) if self.fixed_beaker_offset is not None
                  else lambda m: self.rng.uniform(-self.beaker_range, self.beaker_range, size=(m, 2)))
        draw_d = (fixed(self.fixed_dish_offset) if self.fixed_dish_offset is not None
                  else lambda m: self.rng.uniform(-self.dish_range, self.dish_range, size=(m, 2)))
        b, d = draw_b(k), draw_d(k)
        if self.fixed_beaker_offset is not None and self.fixed_dish_offset is not None:
            bp, dp = b_nom + b[0], d_nom + d[0]
            if min(np.linalg.norm(bp), np.linalg.norm(dp)) < MIN_REACH_M or np.linalg.norm(bp - dp) < MIN_SEPARATION_M:
                raise SystemExit(f"Invalid layout: beaker {bp.round(3).tolist()}, dish {dp.round(3).tolist()} "
                                 f"(each must be >= {MIN_REACH_M} m from the robot base and >= "
                                 f"{MIN_SEPARATION_M} m apart)")
        for _ in range(100):
            bad = (np.linalg.norm(b_nom + b, axis=1) < MIN_REACH_M) | (np.linalg.norm(d_nom + d, axis=1) < MIN_REACH_M)
            bad |= np.linalg.norm((b_nom + b) - (d_nom + d), axis=1) < MIN_SEPARATION_M
            if not bad.any(): break
            nb = int(bad.sum()); b[bad], d[bad] = draw_b(nb), draw_d(nb)
        else: raise RuntimeError("could not sample valid container positions")
        return b, d

    def _start_waypoints(self, stage, beaker, dish):
        """Tip waypoints that place a reset at the beginning of `stage` without crossing a wall."""
        jitter = lambda r: np.append(self.rng.uniform(-r, r, size=2), 0.0)
        b_high = np.array([beaker[0], beaker[1], CLEAR_Z]); d_high = np.array([dish[0], dish[1], CLEAR_Z])
        if stage == 1: return [b_high + jitter(0.01)]
        if stage == 2: return [b_high, np.array([beaker[0], beaker[1], ASPIRATE_GOAL_Z]) + jitter(0.005)]
        if stage == 3: return [b_high + jitter(0.01)]
        if stage == 4: return [d_high + jitter(0.01)]
        if stage == 5: return [d_high, np.array([dish[0], dish[1], DISPENSE_GOAL_Z]) + jitter(0.005)]
        return []

    def _log_episodes(self, ids):
        log = {}
        for k in self.reward_sums:
            log[f"Episode_Reward/{k}"] = float(self.reward_sums[k][ids].mean() / self.episode_length_s)
        for k, v in self.terms_done.items():
            log[f"Episode_Termination/{k}"] = float(np.count_nonzero(v[ids]))
        success = self.terms_done["success"][ids]
        for s, e in self.skills.tolist():
            mask = (self.start_stage[ids] == s) & (self.end_stage[ids] == e)
            if mask.any():
                log[f"Metrics/success_skill_{s}_to_{e}"] = float(success[mask].mean())
        log["Metrics/max_stage_reached"] = float(self.max_stage_this_ep[ids].mean())
        log["Metrics/safety/penetration_count"] = float(self.penetration_count_this_ep[ids].mean())
        log["Metrics/safety/max_contact_force_n"] = float(self.max_contact_force_this_ep[ids].mean())
        log["Metrics/safety/in_warmup"] = float(self._in_warmup())
        self.extras["log"] = log

    def _reset_idx(self, ids):
        if len(ids) == 0: return
        self._log_episodes(ids)
        for k in self.reward_sums:
            self.reward_sums[k][ids] = 0.0

        skill = self.skills[self.rng.choice(len(self.skills), size=len(ids), p=self.skill_probs)]
        self.start_stage[ids], self.end_stage[ids] = skill[:, 0], skill[:, 1]
        b_off, d_off = self._sample_offsets(len(ids))
        self.beaker_offset[ids] = b_off; self.dish_offset[ids] = d_off
        noise = self.rng.uniform(-RESET_JOINT_NOISE, RESET_JOINT_NOISE, size=(len(ids), 9))
        for j, i in enumerate(ids):
            d = self.datas[i]
            start = int(self.start_stage[i])
            qpos = np.clip(HOME_QPOS + (noise[j] if start == 0 else 0.0), self.joint_range[:, 0], self.joint_range[:, 1])
            d.qpos[:9] = qpos; d.qvel[:] = 0.0
            d.ctrl[:7] = qpos[:7]; d.ctrl[7] = 255.0
            d.mocap_pos[self.beaker_mocap] = np.array(BEAKER_POS) + np.array([b_off[j, 0], b_off[j, 1], 0.0])
            d.mocap_pos[self.dish_mocap] = np.array(DISH_POS) + np.array([d_off[j, 0], d_off[j, 1], 0.0])
            mujoco.mj_forward(self.model, d)
            waypoints = self._start_waypoints(start, d.mocap_pos[self.beaker_mocap], d.mocap_pos[self.dish_mocap])
            for wp in waypoints:
                self._settle_tip_target(i, wp)
            # Synchronize the Cartesian setpoint with the physical reset pose.
            self.target_pos[i] = d.site_xpos[self.tip_site] if waypoints else HOME_TIP_POS

        start = self.start_stage[ids]
        carrying = (start >= 2) & (start <= 4)
        self.target_ml[ids] = TARGET_ML
        self.pipette_ml[ids] = np.where(carrying, TARGET_ML, 0.0)
        self.dish_ml[ids] = np.where(start == 5, TARGET_ML, 0.0)
        self.beaker_ml[ids] = BEAKER_INITIAL_ML - np.where(start >= 2, TARGET_ML, 0.0)
        self.prev_pipette_ml[ids] = self.pipette_ml[ids]; self.prev_dish_ml[ids] = self.dish_ml[ids]

        self.task_stage[ids] = start
        self.max_stage_this_ep[ids] = start
        self.stage_advanced_this_step[ids] = False
        self.action[ids] = 0.0; self.prev_action[ids] = 0.0
        self.episode_length_buf[torch.as_tensor(ids)] = 0
        self._refresh_poses(ids)
        self.prev_tip_pos[ids] = self.tip_pos[ids]
        self.contact_force_this_step[ids] = 0.0
        self.penetration_event_this_step[ids] = False
        self.penetration_count_this_ep[ids] = 0
        self.max_contact_force_this_ep[ids] = 0.0
        self.prev_dist_beaker[ids] = np.linalg.norm(self.tip_pos[ids, :2] - self.beaker_pos[ids, :2], axis=1)
        self.prev_tip_z[ids] = self.tip_pos[ids, 2]

    def reset(self):
        self._reset_idx(np.arange(self.num_envs))
        self.extras = {}
        return self._compute_obs(), self.extras

    # ---------------------------------------------------------------- step
    def step(self, actions):
        a = actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions)
        self.prev_action[:] = self.action
        self.action[:] = np.clip(a, -1.0, 1.0)
        # The policy's action is applied as-is: no per-stage override, masking, or setpoint snap.
        self.prev_tip_pos[:] = self.tip_pos
        self.prev_pipette_ml[:] = self.pipette_ml; self.prev_dish_ml[:] = self.dish_ml
        self.target_pos += self.action * STEP_SIZE
        self.target_pos = self.tip_pos + np.clip(self.target_pos - self.tip_pos, -TARGET_LEASH_M, TARGET_LEASH_M)
        np.clip(self.target_pos, self.ws_min, self.ws_max, out=self.target_pos)
        self._apply_actions()
        if self.viewer is not None:
            self.viewer.sync()

        self.episode_length_buf += 1
        self.total_env_steps += self.num_envs
        self._refresh_poses()
        self._update_contact_forces()
        self._update_penetration()
        self._update_fluid()

        # Progress toward the waypoint of the stage that was active during this action.
        stage = self.task_stage.copy()
        goal = self._stage_goal(stage)
        self.step_progress = (np.linalg.norm(self.prev_tip_pos - goal, axis=1)
                              - np.linalg.norm(self.tip_pos - goal, axis=1))
        self._advance_stage()
        self._compute_terminations()
        reward = self._compute_rewards(stage)

        self.extras = {}
        self._done_max_stage = self.max_stage_this_ep.copy()  # read by eval stats after auto-reset
        if self.auto_reset:
            self._reset_idx(np.nonzero(self.terminated | self.time_outs)[0])
        obs = self._compute_obs()
        t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=self.device)
        return obs, t(reward, torch.float32), t(self.terminated, torch.bool), t(self.time_outs, torch.bool), self.extras

    def launch_viewer(self):
        import mujoco.viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.datas[0])
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.viewer.cam.fixedcamid = self.model.cam("viewer").id
        return self.viewer

    def viewer_running(self): return self.viewer is None or self.viewer.is_running()
    def close(self):
        if self.viewer is not None: self.viewer.close(); self.viewer = None
        self.pool.shutdown()


class RslRlVecEnv:
    def __init__(self, env, clip_actions=1.0):
        self.env = env; self.unwrapped = env; self.num_envs = env.num_envs
        self.num_actions = 3; self.max_episode_length = env.max_episode_length
        self.device = env.device; self.clip_actions = clip_actions
        self.cfg = {"task": "liquid_transfer_v10_mujoco", "num_envs": env.num_envs, "episode_length_s": env.episode_length_s}

    @property
    def episode_length_buf(self): return self.env.episode_length_buf
    @episode_length_buf.setter
    def episode_length_buf(self, v): self.env.episode_length_buf = v.cpu()

    def get_observations(self):
        from tensordict import TensorDict
        return TensorDict(self.env.obs_buf, batch_size=[self.num_envs])

    def step(self, actions):
        from tensordict import TensorDict
        actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs, rew, terminated, truncated, extras = self.env.step(actions)
        extras["time_outs"] = truncated
        return TensorDict(obs, batch_size=[self.num_envs]), rew, (terminated | truncated).to(torch.long), extras

    def close(self): self.env.close()


def make_ppo_cfg(seed, device, max_iterations, run_name):
    mlp = lambda dist: {"class_name": "MLPModel", "hidden_dims": [256, 256, 128], "activation": "elu",
                        "obs_normalization": False, "distribution_cfg": dist}  # obs are pre-scaled
    return {
        "seed": seed, "device": device, "num_steps_per_env": 32,
        "init_at_random_ep_len": True, "max_iterations": max_iterations,
        "empirical_normalization": {},
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "clip_actions": 1.0, "check_for_nan": True, "save_interval": 50,
        "experiment_name": "liquid_transfer_v10", "run_name": run_name,
        "logger": "tensorboard", "resume": False, "load_run": ".*",
        "load_checkpoint": "model_.*.pt", "class_name": "OnPolicyRunner",
        "actor": mlp({"class_name": "GaussianDistribution", "init_std": 0.6, "std_type": "scalar"}),
        "critic": mlp(None),
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 5, "num_mini_batches": 4,
                      "learning_rate": 3.0e-4, "schedule": "adaptive", "gamma": 0.99, "lam": 0.95,
                      "entropy_coef": 0.005, "desired_kl": 0.01, "max_grad_norm": 1.0,
                      "optimizer": "adam", "value_loss_coef": 1.0, "use_clipped_value_loss": True,
                      "clip_param": 0.2, "normalize_advantage_per_mini_batch": False,
                      "share_cnn_encoders": False, "use_mixed_precision": False,
                      "rnd_cfg": None, "symmetry_cfg": None},
        "torch_compile_mode": None, "policy": {},
    }


def _pace(t_last):
    if args_cli.viz: time.sleep(max(0.0, t_last + 1.0 / CONTROL_HZ - time.perf_counter()))
    return time.perf_counter()


class EpisodeStats:
    """Outcome counts, success per distance bin, and the furthest stage reached by failures."""
    def __init__(self):
        self.counts = {}; self.per_bin = {"dish": {}, "beaker": {}}; self.ep_lens = []; self.fail_stage = {}

    def record(self, env, done_ids, ep_len, dish_off, beaker_off, max_stage):
        succ = env.terms_done["success"]
        for i in done_ids.tolist():
            outcome = next((k for k in ("success", "integrity_violation", "excessive_contact_force",
                                        "tip_below_table", "time_out") if env.terms_done[k][i]), "skill_over")
            self.counts[outcome] = self.counts.get(outcome, 0) + 1
            if not succ[i]:
                self.fail_stage[int(max_stage[i])] = self.fail_stage.get(int(max_stage[i]), 0) + 1
            self.ep_lens.append(ep_len[i])
        for name, off in (("dish", dish_off), ("beaker", beaker_off)):
            bins = distance_bin(np.linalg.norm(off, axis=1))
            for i in done_ids.tolist():
                st = self.per_bin[name].setdefault(int(bins[i]), [0, 0])
                st[0] += int(succ[i]); st[1] += 1

    def report(self, tag, steps, n, elapsed):
        episodes = sum(self.counts.values())
        rate = self.counts.get("success", 0) / max(episodes, 1)
        mean_len = sum(self.ep_lens) / max(len(self.ep_lens), 1)
        print(f"[{tag}] {steps} steps x {n} envs: {episodes} eps, outcomes {self.counts}, "
              f"success {rate:.1%}, mean ep len {mean_len:.0f}; {elapsed:.1f}s")
        if self.fail_stage:
            print(f"[{tag}] failures by furthest stage reached: {dict(sorted(self.fail_stage.items()))}")
        for name in ("dish", "beaker"):
            for b in sorted(self.per_bin[name]):
                s, k = self.per_bin[name][b]
                print(f"[{tag}] {name} {distance_bin_label(b)}: {s}/{k} = {s/max(k,1):.1%}")


def run_episodes(env, act_fn, max_steps, tag, trace=False):
    obs, _ = env.reset(); n = env.num_envs
    stats = EpisodeStats(); ep_len = np.zeros(n)
    t = t0 = time.perf_counter()
    for step in range(max_steps):
        if not env.viewer_running(): break
        dish_before, beaker_before = env.dish_offset.copy(), env.beaker_offset.copy()
        stage_before = env.task_stage[0]
        actions = act_fn(obs)
        obs, _, terminated, truncated, _ = env.step(actions)
        ep_len += 1
        done_ids = np.nonzero((terminated | truncated).cpu().numpy())[0]
        if trace and (step % 30 == 0 or env.stage_advanced_this_step[0] or 0 in done_ids):
            a0 = np.clip(np.asarray(actions[0].cpu() if isinstance(actions, torch.Tensor) else actions[0]), -1, 1)
            print(f"[TRACE] t={int(ep_len[0]):3d} stage={stage_before} applied={np.round(a0, 3)}")
        if len(done_ids):
            stats.record(env, done_ids, ep_len, dish_before, beaker_before, env._done_max_stage)
            ep_len[done_ids] = 0
        t = _pace(t)
    stats.report(tag, max_steps, n, time.perf_counter() - t0)


def _load_policy_from_checkpoint(checkpoint_path, device="cpu"):
    """Load actor MLP + obs normalizer from an rsl_rl checkpoint. Returns a policy(obs)->actions callable."""
    import torch.nn as nn
    ckpt = torch.load(checkpoint_path, map_location=device)
    actor_sd = ckpt.get("actor_state_dict", ckpt)
    mlp_keys = sorted([k for k in actor_sd if k.startswith("mlp.") and k.endswith(".weight")],
                      key=lambda s: int(s.split(".")[1]))
    if not mlp_keys:
        raise RuntimeError(f"no mlp.*.weight keys found in checkpoint: {list(actor_sd.keys())}")
    layers, prev = [], actor_sd[mlp_keys[0]].shape[1]
    in_dim = prev
    for i, k in enumerate(mlp_keys):
        layers.append(nn.Linear(prev, actor_sd[k].shape[0])); prev = actor_sd[k].shape[0]
        if i < len(mlp_keys) - 1:
            layers.append(nn.ELU())
    net = nn.Sequential(*layers).to(device)
    net.load_state_dict({k.split(".", 1)[1]: v for k, v in actor_sd.items() if k.startswith("mlp.")})
    net.eval()
    if "obs_normalizer._mean" in actor_sd:
        obs_mean, obs_std = actor_sd["obs_normalizer._mean"].to(device), actor_sd["obs_normalizer._std"].to(device)
    else:
        obs_mean, obs_std = torch.zeros(in_dim, device=device), torch.ones(in_dim, device=device)
    print(f"[Policy] loaded {checkpoint_path}: in_dim={in_dim}, "
          f"hidden={[l.out_features for l in layers if isinstance(l, nn.Linear)][:-1]}")

    def policy(obs):
        with torch.no_grad():
            return net((obs - obs_mean) / (obs_std + 1e-8))
    return policy


def run_play(env, checkpoint_path, max_steps):
    policy = _load_policy_from_checkpoint(checkpoint_path, env.device)
    override = args_cli.action_override
    def act(obs):
        a = policy(obs["policy"])
        if override == "zeros": return torch.zeros_like(a)
        if override == "random": return torch.empty_like(a).uniform_(-1.0, 1.0)
        return a
    run_episodes(env, act, max_steps, f"Play action_override={override}", trace=args_cli.trace)


def run_scripted(env, max_steps):
    """Baseline outside the environment: proportional control toward the stage waypoint."""
    def act(_obs):
        goal = env._stage_goal(env.task_stage)
        a = np.clip((goal - env.target_pos) / STEP_SIZE, -DESCENT_MAX_ACTION, DESCENT_MAX_ACTION)
        vertical = np.isin(env.task_stage, (1, 2, 4, 5)) & (np.linalg.norm(goal[:, :2] - env.tip_pos[:, :2], axis=1) < XY_TOL_M)
        a[vertical, :2] *= 0.5
        return a
    run_episodes(env, act, max_steps, "Scripted", trace=args_cli.trace)


def run_train(env, agent_cfg, log_dir, resume_path, phase):
    from rsl_rl.runners import OnPolicyRunner
    runner = OnPolicyRunner(RslRlVecEnv(env, agent_cfg["clip_actions"]), agent_cfg, log_dir=log_dir, device=agent_cfg["device"])
    if resume_path:
        runner.load(resume_path, map_location=agent_cfg["device"])
        # Action std shrinks as a phase converges; a new skill needs exploration again,
        # otherwise the warm-started policy stays stuck in the previous skill's behavior.
        for mod in vars(runner.alg).values():
            if isinstance(mod, torch.nn.Module):
                for pname, p in mod.named_parameters():
                    if pname.endswith("std_param"):
                        old = p.data.clone(); p.data.clamp_(min=args_cli.warm_start_std)
                        print(f"[Train] {pname}: {old.numpy().round(3).tolist()} -> {p.data.numpy().round(3).tolist()}")
        print(f"[Train] warm start from {resume_path}")
    print(f"[Train] phase={phase} skills={env.skills.tolist()} probs={env.skill_probs.round(3).tolist()} "
          f"warmup_steps={env.warmup_steps}, logging to {log_dir}")
    runner.learn(num_learning_iterations=agent_cfg["max_iterations"], init_at_random_ep_len=True)
    os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)
    out = os.path.join(HERE, "checkpoints", f"{phase}.pt")
    runner.save(out)
    print(f"[Train] saved {out}")


def _parse_offset(s, what):
    if not s: return None
    try:
        dx, dy = (float(v) for v in s.split(","))
    except ValueError:
        raise SystemExit(f"--{what}_offset must be 'dx,dy' in metres, got {s!r}")
    return (dx, dy)


def main():
    device = args_cli.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[Device] CUDA unavailable, using cpu"); device = "cpu"
    num_steps_per_env = 32  # must match make_ppo_cfg
    warmup = int(args_cli.max_iterations * num_steps_per_env * args_cli.num_envs * args_cli.warmup_frac) if args_cli.mode == "train" else 0
    env = LiquidTransferV10Env(
        num_envs=args_cli.num_envs, phase=args_cli.phase, seed=args_cli.seed,
        episode_length_s=args_cli.episode_length_s, dish_range=args_cli.dish_range,
        beaker_range=args_cli.beaker_range, device=device, threads=args_cli.threads,
        fixed_beaker_offset=_parse_offset(args_cli.beaker_offset, "beaker"),
        fixed_dish_offset=_parse_offset(args_cli.dish_offset, "dish"),
        warmup_steps=warmup,
    )
    if args_cli.viz: env.launch_viewer()
    print(f"*** Policy 10 (MuJoCo {mujoco.__version__}): {args_cli.num_envs} envs, mode={args_cli.mode}, "
          f"phase={args_cli.phase}, "
          f"beaker={'offset ' + args_cli.beaker_offset if args_cli.beaker_offset else f'±{args_cli.beaker_range:.3f} m'}, "
          f"dish={'offset ' + args_cli.dish_offset if args_cli.dish_offset else f'±{args_cli.dish_range:.3f} m'}, "
          f"episode={args_cli.episode_length_s:.0f}s, obs={tuple(env.obs_buf['policy'].shape)} ***")
    try:
        if args_cli.mode == "scripted":
            run_scripted(env, args_cli.max_steps)
        elif args_cli.mode == "train":
            torch.manual_seed(args_cli.seed)
            run_name = args_cli.run_name or args_cli.phase
            cfg = make_ppo_cfg(args_cli.seed, device, args_cli.max_iterations, run_name)
            name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{run_name}"
            run_train(env, cfg, os.path.join(HERE, "logs", "rsl_rl", cfg["experiment_name"], name),
                      args_cli.checkpoint, args_cli.phase)
        else:
            if not args_cli.checkpoint: raise SystemExit("play needs --checkpoint <.pt>")
            run_play(env, args_cli.checkpoint, args_cli.max_steps)
    finally:
        env.close()

if __name__ == "__main__":
    main()
