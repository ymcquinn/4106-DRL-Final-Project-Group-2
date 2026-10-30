#liquid_transfer_m3_v6.py

"""
Model 3 (MuJoCo): PPO with known container positions, both beaker and dish random every episode.

REWARD VERSIONS (--reward_version):
  v4 : original. 6 terms, no safety. Learns to slide the pipette through the beaker wall
       at low altitude. High "success", zero integrity.
  v6 : v4 + contact-force penalty + penetration detection + capped safe-transit shaping
       + success integrity gating. Warmup period softens termination so exploration can
       find the lift behavior before the hard constraint engages.

Run from this folder with a venv that has mujoco, torch, rsl_rl, tensorboard:
    bash setup_env.sh --cpu

    Probe:      .venv/bin/python liquid_transfer_m3_v6.py --mode probe --num_envs 2 --viz
    Scripted:   .venv/bin/python liquid_transfer_m3_v6.py --mode scripted --num_envs 16 --max_steps 900
    Train v6:   .venv/bin/python liquid_transfer_m3_v6.py --mode train --num_envs 64 \
                    --max_iterations 500 --reward_version v6
    Eval:       .venv/bin/python liquid_transfer_m3_v6.py --mode play --num_envs 32 \
                    --max_steps 1800 --seed 7 --checkpoint <run>/model_500.pt --reward_version v6
"""

#from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

parser = argparse.ArgumentParser(description="Liquid transfer, model 3 in MuJoCo.")
parser.add_argument("--mode", choices=["random", "probe", "scripted", "train", "play"], default="random")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--max_steps", type=int, default=300)
parser.add_argument("--max_iterations", type=int, default=300)
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--episode_length_s", type=float, default=15.0)
parser.add_argument("--dish_range", type=float, default=0.20)
parser.add_argument("--beaker_range", type=float, default=0.20)
parser.add_argument("--dish_offset", type=str, default="")
parser.add_argument("--beaker_offset", type=str, default="")
parser.add_argument("--hold", action="store_true")
parser.add_argument(
    "--diagnose_timeouts",
    action="store_true",
    help=(
        "Print the final state of every timed-out episode before automatic reset."
    ),
)
parser.add_argument("--viz", action="store_true")
parser.add_argument(
    "--record_mp4",
    type=str,
    default="",
    help="Write env 0 to an MP4 file. Recording stops at the first episode outcome.",
)
parser.add_argument("--record_width", type=int, default=1280)
parser.add_argument("--record_height", type=int, default=720)
parser.add_argument("--record_fps", type=int, default=30)
parser.add_argument(
    "--record_camera",
    type=str,
    default="viewer",
    help="MJCF fixed-camera name used for MP4 recording.",
)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--threads", type=int, default=0)
parser.add_argument("--reward_version", choices=["v4", "v6"], default="v6")
parser.add_argument(
    "--curriculum",
    choices=["full", "center_above", "aspirate", "lift", "center_lift", "approach_center_lift", "transit_to_dish", "dispense_lift", "end_to_end"],
    default="full",
    help=(
        "Training/evaluation task. 'center_lift' teaches fixed-beaker centering, "
        "controlled aspiration, and vertical lift above the rim before dish transit."
    ),
)
parser.add_argument("--warmup_frac", type=float, default=0.20,
                    help="Fraction of training steps during which penetration is penalized but not terminating.")
args_cli = parser.parse_args()

REWARD_VERSION = args_cli.reward_version
WARMUP_FRAC = args_cli.warmup_frac

if not args_cli.viz:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(HERE, "liquid_transfer_scene.xml")

# Geometry
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
TARGET_RANGE_ML = (2.0, 10.0)
SUCCESS_TOL_ML = 0.3
ASPIRATE_MAX_TIP_Z = 0.11
DISPENSE_MAX_TIP_Z = 0.10

# Control
CONTROL_HZ = 30.0
DECIMATION = 4
PHYSICS_SUBSTEPS = 2
STEP_SIZE = 0.01
HOME_TIP_POS = (0.28, 0.0, 0.38)
WORKSPACE_MIN = (0.15, -0.45, 0.03)
WORKSPACE_MAX = (0.65, 0.45, 0.45)
IK_LAMBDA = 0.01
HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
RESET_JOINT_NOISE = 0.1

# Reward v4
REWARD_PROGRESS_PER_ML = 10.0
REWARD_SUCCESS = 100.0
REWARD_REACH_PER_M = -1.0
REWARD_TIME = -0.05
REWARD_OVERFILL = -20.0
REWARD_ACTION_RATE = -0.01
# Validated fixed-beaker reset pose for the isolated aspiration curriculum.
ASPIRATE_START_QPOS = np.array([
    -0.169280472,
    -0.366600519,
    -0.242293158,
    -2.387573410,
    -0.096437487,
     2.025275763,
     0.434830576,
     0.039995138,
     0.039998882,
])
# Validated fixed-beaker reset pose for the isolated aspiration curriculum.
ASPIRATE_START_TIP_POS = np.array([
     0.394540439,
    -0.181969082,
     0.253299549,
])

# Reward v6 additions
REWARD_CONTACT_FORCE_PER_N = -0.5
REWARD_CONTACT_FORCE_THRESHOLD_N = 5.0
REWARD_CONTACT_FORCE_CAP_N = 100.0
REWARD_PENETRATION_DENSE = -5.0
REWARD_PENETRATION_TERMINATE = -50.0
REWARD_SAFE_TRANSIT = 5.0
SAFE_TRANSIT_MIN_Z = BEAKER_RIM_Z + 0.02      # 2 cm above the rim
SAFE_TRANSIT_MIN_ML = 0.2                      # "carrying" = pipette has at least this much
SAFE_TRANSIT_MIN_XY_MOVE = 0.001               # bonus only if the tip moved ≥1 mm horizontally
SAFE_TRANSIT_CAP_PER_EP = 50.0                 # hard cap so hovering can't farm the reward

# Center-and-lift curriculum.
CENTER_ABOVE_Z = 0.25                         # safe waypoint used by scripted controller
CENTER_ABOVE_XY_TOL_M = 0.020                 # 2 cm horizontal tolerance
CENTER_ABOVE_Z_TOL_M = 0.020                  # 2 cm vertical tolerance
CENTER_ABOVE_REWARD_PROGRESS_PER_M = 80.0     # reward decrease in 3D waypoint error
CENTER_ABOVE_REWARD_PROXIMITY = 1.0           # bounded per-step proximity reward
CENTER_ABOVE_REWARD_SUCCESS = 150.0
ASPIRATE_TARGET_ML = 2.0                    # begin with the easiest requested volume
ASPIRATE_GOAL_Z = 0.08                       # safely inside the beaker
ASPIRATE_XY_TOL_M = 0.020                    # stay near the centerline
ASPIRATE_REWARD_PROGRESS_PER_M = 80.0
ASPIRATE_REWARD_PROXIMITY = 1.0
ASPIRATE_REWARD_PER_ML = 30.0
ASPIRATE_REWARD_SUCCESS = 150.0
ASPIRATE_PENALTY_LATERAL = -2.0
ASPIRATE_DESCENT_MAX_ACTION = 0.35
ASPIRATE_PENALTY_FAST_DESCENT = -1.0

# Controlled-lift curriculum: start centered inside with 2 mL, then lift vertically clear.
LIFT_START_Z = ASPIRATE_GOAL_Z
LIFT_CLEAR_Z = BEAKER_RIM_Z + 0.08
LIFT_XY_TOL_M = 0.020
LIFT_TARGET_ML = ASPIRATE_TARGET_ML
LIFT_RESET_SETTLE_STEPS = 80
LIFT_REWARD_PROGRESS_PER_M = 100.0
LIFT_REWARD_UP_ACTION = 0.25
LIFT_PENALTY_DOWN_ACTION = -0.50
LIFT_REWARD_SUCCESS = 150.0
LIFT_EMERGENCY_FORCE_N = 30.0
CENTER_LIFT_CLEAR_Z = BEAKER_RIM_Z + 0.08       # tip 8 cm above rim before stage success
CENTER_LIFT_CENTER_RADIUS_M = 0.020             # central 2 cm target region
CENTER_LIFT_DESCENT_MAX_ACTION = 0.35           # discourage fast descent near/inside beaker
CENTER_LIFT_REWARD_CENTER_PER_M = 40.0          # reward reduction in XY center error
CENTER_LIFT_REWARD_ASPIRATE_PER_ML = 15.0       # reward liquid entering pipette
CENTER_LIFT_REWARD_VERTICAL_PER_M = 60.0        # reward correct down/up vertical progress
CENTER_LIFT_PENALTY_LATERAL_BELOW_RIM = -2.0    # penalize XY action while below clearance
CENTER_LIFT_PENALTY_FAST_DESCENT = -1.0         # penalize excessive downward command
CENTER_LIFT_REWARD_STAGE_SUCCESS = 150.0
CENTER_LIFT_EMERGENCY_FORCE_N = 30.0            # hard stop; 5 N still invalidates success

REWARD_TERMS_V4 = ("progress", "success", "reach", "time", "overfill", "action_rate")
REWARD_TERMS_V6 = REWARD_TERMS_V4 + (
    "contact_force", "penetration_dense", "penetration_term", "safe_transit",
    "curriculum_center", "curriculum_aspirate", "curriculum_vertical",
    "curriculum_lateral", "curriculum_fast_descent", "curriculum_success",
)
TERM_V4 = (
    "time_out",
    "success",
    "overfill",
    "tip_below_table",
)

TERM_V6 = TERM_V4 + (
    "integrity_violation",
    "excessive_contact_force",
)

# Home-to-center plus aspirate-and-lift curriculum.
APPROACH_CENTER_Z = CENTER_ABOVE_Z
APPROACH_CENTER_XY_TOL_M = CENTER_ABOVE_XY_TOL_M
APPROACH_CENTER_Z_TOL_M = CENTER_ABOVE_Z_TOL_M

APPROACH_REWARD_PROGRESS_PER_M = 80.0
APPROACH_REWARD_PROXIMITY = 1.0
APPROACH_PENALTY_DESCENT_WHILE_OFF_CENTER = -1.0
# Isolated safe transit: carry 2 mL horizontally from above beaker to above dish.
TRANSIT_Z = CENTER_LIFT_CLEAR_Z
TRANSIT_XY_TOL_M = 0.020
TRANSIT_Z_TOL_M = 0.020
TRANSIT_TARGET_ML = ASPIRATE_TARGET_ML
TRANSIT_REWARD_PROGRESS_PER_M = 100.0
TRANSIT_REWARD_PROXIMITY = 1.0
TRANSIT_REWARD_SUCCESS = 150.0
# Isolated dish dispensing and extraction.
DISPENSE_START_Z = TRANSIT_Z
DISPENSE_GOAL_Z = 0.06
DISPENSE_CLEAR_Z = TRANSIT_Z
DISPENSE_XY_TOL_M = 0.020
DISPENSE_TARGET_ML = ASPIRATE_TARGET_ML
DISPENSE_REWARD_PROGRESS_PER_M = 80.0
DISPENSE_REWARD_PER_ML = 30.0
DISPENSE_REWARD_VERTICAL_PER_M = 60.0
DISPENSE_REWARD_SUCCESS = 150.0
DISPENSE_DESCENT_MAX_ACTION = 0.35
DISPENSE_PENALTY_FAST_DESCENT = -1.0
# End-to-end six-stage state machine.
E2E_NUM_STAGES = 6
E2E_STAGE_BONUS = 25.0

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


class LiquidTransferM3Env:
    def __init__(self, num_envs, seed=42, episode_length_s=15.0, dish_range=0.20, beaker_range=0.20,
                 fixed_dish_offset=None, fixed_beaker_offset=None, device="cpu", threads=0, auto_reset=True,
                 total_training_steps=0, reward_version="v6", warmup_frac=0.20, curriculum="full"):
        self.num_envs = num_envs
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.dish_range, self.beaker_range = dish_range, beaker_range
        self.fixed_dish_offset, self.fixed_beaker_offset = fixed_dish_offset, fixed_beaker_offset
        self.auto_reset = auto_reset
        self.step_dt = 1.0 / CONTROL_HZ
        self.episode_length_s = episode_length_s
        self.max_episode_length = math.ceil(episode_length_s / self.step_dt)
        self.reward_version = reward_version
        self.curriculum = curriculum
        self.warmup_steps = int(total_training_steps * warmup_frac)
        self.total_env_steps = 0

        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        assert abs(self.model.opt.timestep * PHYSICS_SUBSTEPS * DECIMATION - self.step_dt) < 1e-6
        self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)]
        m = self.model
        self.hand_id = m.body("hand").id
        self.tip_site = m.site("pipette_tip").id
        self.beaker_geom_ids = {
            m.geom("beaker_geom").id,
            *{
                m.geom(f"beaker_wall_{i:02d}").id
                for i in range(16)
            },
        }
        self.dish_geom_ids = {
            m.geom("dish_geom").id,
        }
        self.beaker_mocap = m.body_mocapid[m.body("beaker").id]
        self.dish_mocap = m.body_mocapid[m.body("petri_dish").id]
        self.joint_range = m.jnt_range.copy()
        assert np.allclose(m.key_qpos[m.key("home").id], HOME_QPOS)

        n = num_envs
        self.target_pos = np.tile(np.array(HOME_TIP_POS), (n, 1))
        self.ws_min, self.ws_max = np.array(WORKSPACE_MIN), np.array(WORKSPACE_MAX)
        self.action = np.zeros((n, 3))
        self.prev_action = np.zeros((n, 3))
        self._jacp = np.zeros((3, m.nv)); self._jacr = np.zeros((3, m.nv))
        self.target_ml = np.zeros(n); self.pipette_ml = np.zeros(n)
        self.dish_ml = np.zeros(n); self.beaker_ml = np.full(n, BEAKER_INITIAL_ML)
        self.progress = np.zeros(n)
        self.tip_pos = np.zeros((n, 3)); self.beaker_pos = np.zeros((n, 3)); self.dish_pos = np.zeros((n, 3))
        self.dish_offset = np.zeros((n, 2)); self.beaker_offset = np.zeros((n, 2))
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)

        # v6 safety + safe-transit state
        self.prev_dist_beaker = np.zeros(n)
        self.prev_tip_z = np.zeros(n)
        self.prev_tip_xy = np.zeros((n, 2))
        self.contact_force_this_step = np.zeros(n)
        self.penetration_event_this_step = np.zeros(n, dtype=bool)
        self.penetration_count_this_ep = np.zeros(n, dtype=int)
        self.max_contact_force_this_ep = np.zeros(n)
        self.safe_transit_this_ep = np.zeros(n)
        self.prev_center_above_distance = np.zeros(n)
        self.prev_beaker_xy_dist = np.zeros(n)
        self.prev_curriculum_tip_z = np.zeros(n)
        self.prev_curriculum_pipette_ml = np.zeros(n)
        # One-way per-episode latch for home -> center-above completion.
        self.approach_complete = np.zeros(n, dtype=bool)
        # 0 approach beaker, 1 aspirate, 2 lift beaker, 3 transit,
        # 4 dispense, 5 lift dish.
        self.task_stage = np.zeros(n, dtype=np.int64)
        self.stage_advanced_this_step = np.zeros(n, dtype=bool)

        terms = TERM_V6 if reward_version == "v6" else TERM_V4
        self.terms_done = {k: np.zeros(n, dtype=bool) for k in terms}
        self.terminated = np.zeros(n, dtype=bool)
        self.time_outs = np.zeros(n, dtype=bool)
        rt = REWARD_TERMS_V6 if reward_version == "v6" else REWARD_TERMS_V4
        self.reward_sums = {k: np.zeros(n) for k in rt}
        self.extras = {}; self.obs_buf = {}
        self.pool = ThreadPoolExecutor(max_workers=threads or min(8, os.cpu_count() or 1))
        self.viewer = None
        self.reset()

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

    def _settle_tip_target(self, idx, goal, steps=LIFT_RESET_SETTLE_STEPS):
        """Move one reset state to a Cartesian tip goal before exposing it to PPO."""
        d = self.datas[idx]
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for _ in range(steps):
            mujoco.mj_jacSite(self.model, d, jacp, jacr, self.tip_site)
            jac = np.zeros((6, 7))
            jac[:3] = jacp[:, :7]
            jac[3:] = jacr[:, :7]
            err = np.zeros(6)
            err[:3] = goal - d.site_xpos[self.tip_site]
            cur_quat = d.xquat[self.hand_id][None, :]
            q_err = quat_mul(TIP_DOWN_QUAT_WXYZ[None, :], quat_conj(cur_quat))
            err[3:] = axis_angle_from_quat(q_err)[0]
            jjt = jac @ jac.T + (IK_LAMBDA ** 2) * np.eye(6)
            dq = jac.T @ np.linalg.solve(jjt, err)
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

    def _update_contact_forces(self):
        cf = np.zeros(self.num_envs)
        for i, d in enumerate(self.datas):
            total = 0.0
            for c in range(d.ncon):
                cid = d.contact[c]
                g1, g2 = cid.geom1, cid.geom2
                container_contact = (
                    g1 in self.beaker_geom_ids
                    or g2 in self.beaker_geom_ids
                    or g1 in self.dish_geom_ids
                    or g2 in self.dish_geom_ids
                )

                if container_contact:
                    buf = np.zeros(6)
                    mujoco.mj_contactForce(self.model, d, c, buf)
                    total += abs(buf[0])
            cf[i] = min(total, REWARD_CONTACT_FORCE_CAP_N)
        self.contact_force_this_step = cf
        self.max_contact_force_this_ep = np.maximum(self.max_contact_force_this_ep, cf)

    def _update_penetration(self):
        tip = self.tip_pos
        d_b = np.linalg.norm(tip[:, :2] - self.beaker_pos[:, :2], axis=1)
        z = tip[:, 2]
        prev_in_low = (self.prev_dist_beaker < BEAKER_RADIUS) & (self.prev_tip_z < BEAKER_RIM_Z)
        leave_low = prev_in_low & (d_b >= BEAKER_RADIUS) & (z < BEAKER_RIM_Z)
        self.penetration_event_this_step = leave_low
        self.penetration_count_this_ep += leave_low.astype(int)
        self.prev_dist_beaker = d_b
        self.prev_tip_z = z

    def _update_safety(self):
        if self.reward_version == "v6":
            self._refresh_poses()
            self._update_contact_forces()
            self._update_penetration()

    def _integrity_ok(self):
        if self.reward_version != "v6":
            return np.ones(self.num_envs, dtype=bool)
        return (self.penetration_count_this_ep == 0) & (self.max_contact_force_this_ep < REWARD_CONTACT_FORCE_THRESHOLD_N)

    def _update_fluid(self):
        self._refresh_poses()
        tip = self.tip_pos
        in_beaker = (np.linalg.norm(tip[:, :2] - self.beaker_pos[:, :2], axis=1) < BEAKER_RADIUS - 0.005) & (tip[:, 2] < ASPIRATE_MAX_TIP_Z)
        if self.curriculum in ("center_above", "lift", "transit_to_dish", "dispense_lift"):
            in_beaker = np.zeros(self.num_envs, dtype=bool)
        take = np.where(
            in_beaker,
            FLOW_ML_PER_STEP,
            0.0,
        )

        remaining_capacity = (
            PIPETTE_CAPACITY_ML
            - self.pipette_ml
        )

        if self.curriculum in (
            "aspirate",
            "center_lift",
            "approach_center_lift",
            "end_to_end",
        ):
            # Isolated and combined aspiration stages should stop
            # collecting liquid once the requested amount is reached.
            remaining_capacity = np.minimum(
                remaining_capacity,
                np.maximum(
                    0.0,
                    self.target_ml
                    - self.pipette_ml,
                ),
            )

        take = np.minimum(
            take,
            np.minimum(
                self.beaker_ml,
                remaining_capacity,
            ),
        )
        self.beaker_ml -= take; self.pipette_ml += take
        over_dish = (np.linalg.norm(tip[:, :2] - self.dish_pos[:, :2], axis=1) < DISH_RADIUS - 0.01) & (tip[:, 2] < DISPENSE_MAX_TIP_Z)
        if self.curriculum in ("center_above", "aspirate", "lift", "center_lift", "approach_center_lift", "transit_to_dish"):
            over_dish = np.zeros(self.num_envs, dtype=bool)
        give = np.where(over_dish, FLOW_ML_PER_STEP, 0.0)
        give = np.minimum(give, np.minimum(self.pipette_ml, DISH_CAPACITY_ML - self.dish_ml))
        before = np.minimum(self.dish_ml, self.target_ml)
        self.pipette_ml -= give; self.dish_ml += give
        self.progress = np.minimum(self.dish_ml, self.target_ml) - before

    def _center_above_goal(self):
        goal = self.beaker_pos.copy()
        goal[:, 2] = CENTER_ABOVE_Z
        return goal

    def _center_above_success(self):
        goal = self._center_above_goal()
        xy_ok = np.linalg.norm(self.tip_pos[:, :2] - goal[:, :2], axis=1) <= CENTER_ABOVE_XY_TOL_M
        z_ok = np.abs(self.tip_pos[:, 2] - CENTER_ABOVE_Z) <= CENTER_ABOVE_Z_TOL_M
        return xy_ok & z_ok & self._integrity_ok()

    def _aspirate_goal(self):
        goal = self.beaker_pos.copy()
        goal[:, 2] = ASPIRATE_GOAL_Z
        return goal

    def _aspirate_success(self):
        xy = np.linalg.norm(self.tip_pos[:, :2] - self.beaker_pos[:, :2], axis=1)
        enough = self.pipette_ml >= (self.target_ml - SUCCESS_TOL_ML)
        low_enough = self.tip_pos[:, 2] <= ASPIRATE_MAX_TIP_Z
        return enough & (xy <= ASPIRATE_XY_TOL_M) & low_enough & self._integrity_ok()

    def _lift_goal(self):
        goal = self.beaker_pos.copy()
        goal[:, 2] = LIFT_CLEAR_Z
        return goal

    def _lift_success(self):
        xy = np.linalg.norm(self.tip_pos[:, :2] - self.beaker_pos[:, :2], axis=1)
        enough = self.pipette_ml >= (LIFT_TARGET_ML - SUCCESS_TOL_ML)
        clear = self.tip_pos[:, 2] >= LIFT_CLEAR_Z
        return enough & (xy <= LIFT_XY_TOL_M) & clear & self._integrity_ok()

    def _transit_goal(self):
        goal = self.dish_pos.copy()
        goal[:, 2] = TRANSIT_Z
        return goal
    def _transit_success(self):
        xy = np.linalg.norm(self.tip_pos[:, :2] - self.dish_pos[:, :2], axis=1)
        z_ok = np.abs(self.tip_pos[:, 2] - TRANSIT_Z) <= TRANSIT_Z_TOL_M
        carrying = self.pipette_ml >= (TRANSIT_TARGET_ML - SUCCESS_TOL_ML)
        return (xy <= TRANSIT_XY_TOL_M) & z_ok & carrying & self._integrity_ok()
    def _dispense_goal(self):
        goal = self.dish_pos.copy()
        emptied = self.pipette_ml <= SUCCESS_TOL_ML
        goal[:, 2] = np.where(emptied, DISPENSE_CLEAR_Z, DISPENSE_GOAL_Z)
        return goal
    def _dispense_success(self):
        xy = np.linalg.norm(self.tip_pos[:, :2] - self.dish_pos[:, :2], axis=1)
        fluid_ok = np.abs(self.dish_ml - self.target_ml) < SUCCESS_TOL_ML
        empty = self.pipette_ml <= SUCCESS_TOL_ML
        clear = self.tip_pos[:, 2] >= DISPENSE_CLEAR_Z
        return fluid_ok & empty & clear & (xy <= DISPENSE_XY_TOL_M) & self._integrity_ok()
    def _end_to_end_goal(self):
        goals = np.zeros_like(self.tip_pos)
        b_high = self.beaker_pos.copy(); b_high[:, 2] = CENTER_LIFT_CLEAR_Z
        b_low = self.beaker_pos.copy(); b_low[:, 2] = ASPIRATE_GOAL_Z
        d_high = self.dish_pos.copy(); d_high[:, 2] = TRANSIT_Z
        d_low = self.dish_pos.copy(); d_low[:, 2] = DISPENSE_GOAL_Z
        choices = (b_high, b_low, b_high, d_high, d_low, d_high)
        for stage, goal in enumerate(choices):
            mask = self.task_stage == stage
            goals[mask] = goal[mask]
        return goals
    def _end_to_end_success(self):
        xy = np.linalg.norm(self.tip_pos[:, :2] - self.dish_pos[:, :2], axis=1)
        return ((self.task_stage == 5) & (xy <= DISPENSE_XY_TOL_M)
                & (
                    self.tip_pos[:, 2]
                    >= DISPENSE_CLEAR_Z - TRANSIT_Z_TOL_M
                )
                & (np.abs(self.dish_ml - self.target_ml) < SUCCESS_TOL_ML)
                & (self.pipette_ml <= SUCCESS_TOL_ML) & self._integrity_ok())

    def _advance_end_to_end_stage(self):
        self.stage_advanced_this_step[:] = False

        if self.curriculum != "end_to_end":
            return

        bxy = np.linalg.norm(
            self.tip_pos[:, :2] - self.beaker_pos[:, :2],
            axis=1,
        )
        dxy = np.linalg.norm(
            self.tip_pos[:, :2] - self.dish_pos[:, :2],
            axis=1,
        )
        z = self.tip_pos[:, 2]

        conditions = (
            # Stage 0 -> 1:
            # centered and safely above the beaker.
            (
                (bxy <= CENTER_ABOVE_XY_TOL_M)
                & (
                    z
                    >= CENTER_LIFT_CLEAR_Z
                    - TRANSIT_Z_TOL_M
                )
            ),

            # Stage 1 -> 2:
            # requested liquid has been aspirated.
            (
                self.pipette_ml
                >= self.target_ml - SUCCESS_TOL_ML
            ),

            # Stage 2 -> 3:
            # lifted clear of the beaker.
            (
                (bxy <= LIFT_XY_TOL_M)
                & (
                    z
                    >= CENTER_LIFT_CLEAR_Z
                    - TRANSIT_Z_TOL_M
                )
            ),

            # Stage 3 -> 4:
            # centered above the dish at safe transit height.
            (
                (dxy <= TRANSIT_XY_TOL_M)
                & (
                    np.abs(z - TRANSIT_Z)
                    <= TRANSIT_Z_TOL_M
                )
            ),

            # Stage 4 -> 5:
            # requested amount has been dispensed.
            (
                self.dish_ml
                >= self.target_ml - SUCCESS_TOL_ML
            ),
        )

        # Transitions are one-way and monotonic.
        for stage, condition in enumerate(conditions):
            advance = (
                (self.task_stage == stage)
                & condition
            )

            # Diagnostic: capture the exact Stage 0 -> 1 transition
            # before task_stage is changed.
            if args_cli.diagnose_timeouts and stage == 0:
                transition_ids = np.flatnonzero(advance)

                for i in transition_ids.tolist():
                    bxy_i = float(
                        np.linalg.norm(
                            self.tip_pos[i, :2]
                            - self.beaker_pos[i, :2]
                        )
                    )

                    print(
                        "[Stage0->1] "
                        f"env={i} "
                        f"episode_step={int(self.episode_length_buf[i])} "
                        f"tip={self.tip_pos[i].round(4).tolist()} "
                        f"target_pos={self.target_pos[i].round(4).tolist()} "
                        f"beaker_pos={self.beaker_pos[i].round(4).tolist()} "
                        f"beaker_offset={self.beaker_offset[i].round(4).tolist()} "
                        f"beaker_xy_error_m={bxy_i:.4f} "
                        f"action={self.action[i].round(4).tolist()} "
                        f"pipette_ml={self.pipette_ml[i]:.2f} "
                        f"max_contact_force_n="
                        f"{self.max_contact_force_this_ep[i]:.3f} "
                        f"penetration_count="
                        f"{int(self.penetration_count_this_ep[i])}"
                    )

            # Stage 0 -> 1 handoff:
            # Stage 1 is vertical-only, so synchronize the Cartesian
            # controller's XY target with the beaker before lateral
            # commands are disabled.
            if stage == 0 and np.any(advance):
                self.target_pos[advance, :2] = (
                    self.beaker_pos[advance, :2]
                )

            self.task_stage[advance] = stage + 1
            self.stage_advanced_this_step |= advance
    def _center_lift_success(self):
        """Stage success: requested liquid acquired, centered, lifted safely, no integrity breach."""
        xy = np.linalg.norm(self.tip_pos[:, :2] - self.beaker_pos[:, :2], axis=1)
        enough = self.pipette_ml >= (self.target_ml - SUCCESS_TOL_ML)
        centered = xy <= CENTER_LIFT_CENTER_RADIUS_M
        clear = self.tip_pos[:, 2] >= CENTER_LIFT_CLEAR_Z
        return enough & centered & clear & self._integrity_ok()

    def _task_success(self):
        if self.curriculum == "center_above":
            return self._center_above_success()
        if self.curriculum == "aspirate":
            return self._aspirate_success()
        if self.curriculum == "lift":
            return self._lift_success()
        if self.curriculum == "transit_to_dish":
            return self._transit_success()
        if self.curriculum == "dispense_lift":
            return self._dispense_success()
        if self.curriculum == "end_to_end":
            return self._end_to_end_success()
        if self.curriculum in (
            "center_lift",
            "approach_center_lift",
        ):
            return self._center_lift_success()
        return self.dish_at_target()

    def _center_above_reward_terms(self):
        n = self.num_envs
        zeros = np.zeros(n)
        if self.curriculum != "center_above":
            return {
                "curriculum_center": zeros.copy(),
                "curriculum_vertical": zeros.copy(),
                "curriculum_success": zeros.copy(),
            }

        goal = self._center_above_goal()
        distance = np.linalg.norm(self.tip_pos - goal, axis=1)
        progress = np.clip(self.prev_center_above_distance - distance, -0.02, 0.02)
        proximity = np.exp(-25.0 * distance)
        self.prev_center_above_distance = distance
        return {
            "curriculum_center": (
                CENTER_ABOVE_REWARD_PROGRESS_PER_M * progress
                + CENTER_ABOVE_REWARD_PROXIMITY * proximity
            ),
            "curriculum_vertical": zeros.copy(),
            "curriculum_success": CENTER_ABOVE_REWARD_SUCCESS * self.terms_done["success"],
        }

    def _aspirate_reward_terms(self):
        n = self.num_envs
        zeros = np.zeros(n)
        if self.curriculum != "aspirate":
            return {
                "curriculum_center": zeros.copy(),
                "curriculum_aspirate": zeros.copy(),
                "curriculum_vertical": zeros.copy(),
                "curriculum_lateral": zeros.copy(),
                "curriculum_fast_descent": zeros.copy(),
                "curriculum_success": zeros.copy(),
            }

        goal = self._aspirate_goal()
        distance = np.linalg.norm(self.tip_pos - goal, axis=1)
        progress = np.clip(self.prev_center_above_distance - distance, -0.02, 0.02)
        proximity = np.exp(-30.0 * distance)
        aspirated = np.maximum(0.0, self.pipette_ml - self.prev_curriculum_pipette_ml)
        xy = np.linalg.norm(self.tip_pos[:, :2] - self.beaker_pos[:, :2], axis=1)
        centered = xy <= ASPIRATE_XY_TOL_M
        below_above_waypoint = self.tip_pos[:, 2] < CENTER_ABOVE_Z

        # X/Y actions are masked by step() in the isolated aspirate stage.
        lateral_penalty = np.zeros(
            self.num_envs,
        )

        downward = np.maximum(0.0, -self.action[:, 2])
        descent_excess = np.maximum(0.0, downward - ASPIRATE_DESCENT_MAX_ACTION)
        fast_descent = np.where(
            centered,
            ASPIRATE_PENALTY_FAST_DESCENT * descent_excess ** 2,
            0.0,
        )

        controlled_descent = np.where(
            self.pipette_ml < self.target_ml - SUCCESS_TOL_ML,
            0.25 * np.clip(
                -self.action[:, 2],
                0.0,
                ASPIRATE_DESCENT_MAX_ACTION,
            ),
            0.0,
        )

        self.prev_center_above_distance = distance
        self.prev_curriculum_pipette_ml = self.pipette_ml.copy()
        return {
            "curriculum_center": (
                ASPIRATE_REWARD_PROGRESS_PER_M * progress
                + ASPIRATE_REWARD_PROXIMITY * proximity
            ),
            "curriculum_aspirate": ASPIRATE_REWARD_PER_ML * aspirated,
            "curriculum_vertical": controlled_descent,
            "curriculum_lateral": lateral_penalty,
            "curriculum_fast_descent": fast_descent,
            "curriculum_success": ASPIRATE_REWARD_SUCCESS * self.terms_done["success"],
        }

    def _lift_reward_terms(self):
        n = self.num_envs
        zeros = np.zeros(n)
        if self.curriculum != "lift":
            return {
                "curriculum_center": zeros.copy(),
                "curriculum_aspirate": zeros.copy(),
                "curriculum_vertical": zeros.copy(),
                "curriculum_lateral": zeros.copy(),
                "curriculum_fast_descent": zeros.copy(),
                "curriculum_success": zeros.copy(),
            }
        goal = self._lift_goal()
        distance = np.linalg.norm(self.tip_pos - goal, axis=1)
        progress = np.clip(self.prev_center_above_distance - distance, -0.02, 0.02)
        upward = np.clip(self.action[:, 2], 0.0, 1.0)
        downward = np.clip(-self.action[:, 2], 0.0, 1.0)
        self.prev_center_above_distance = distance
        return {
            "curriculum_center": zeros.copy(),
            "curriculum_aspirate": zeros.copy(),
            "curriculum_vertical": (
                LIFT_REWARD_PROGRESS_PER_M * progress
                + LIFT_REWARD_UP_ACTION * upward
                + LIFT_PENALTY_DOWN_ACTION * downward
            ),
            "curriculum_lateral": zeros.copy(),
            "curriculum_fast_descent": zeros.copy(),
            "curriculum_success": LIFT_REWARD_SUCCESS * self.terms_done["success"],
        }

    def _transit_reward_terms(self):
        n = self.num_envs
        zeros = np.zeros(n)
        if self.curriculum != "transit_to_dish":
            return {
                "curriculum_center": zeros.copy(),
                "curriculum_aspirate": zeros.copy(),
                "curriculum_vertical": zeros.copy(),
                "curriculum_lateral": zeros.copy(),
                "curriculum_fast_descent": zeros.copy(),
                "curriculum_success": zeros.copy(),
            }
        goal = self._transit_goal()
        distance = np.linalg.norm(self.tip_pos - goal, axis=1)
        progress = np.clip(self.prev_center_above_distance - distance, -0.02, 0.02)
        proximity = np.exp(-20.0 * distance)
        self.prev_center_above_distance = distance
        return {
            "curriculum_center": (
                TRANSIT_REWARD_PROGRESS_PER_M * progress
                + TRANSIT_REWARD_PROXIMITY * proximity
            ),
            "curriculum_aspirate": zeros.copy(),
            "curriculum_vertical": zeros.copy(),
            "curriculum_lateral": zeros.copy(),
            "curriculum_fast_descent": zeros.copy(),
            "curriculum_success": TRANSIT_REWARD_SUCCESS * self.terms_done["success"],
        }
    def _dispense_reward_terms(self):
        n = self.num_envs
        zeros = np.zeros(n)
        if self.curriculum != "dispense_lift":
            return {k: zeros.copy() for k in (
                "curriculum_center", "curriculum_aspirate", "curriculum_vertical",
                "curriculum_lateral", "curriculum_fast_descent", "curriculum_success")}
        xy = np.linalg.norm(self.tip_pos[:, :2] - self.dish_pos[:, :2], axis=1)
        centered = xy <= DISPENSE_XY_TOL_M
        emptied = self.pipette_ml <= SUCCESS_TOL_ML
        dz = self.tip_pos[:, 2] - self.prev_curriculum_tip_z
        dispensed = np.maximum(0.0, self.prev_curriculum_pipette_ml - self.pipette_ml)
        descend = np.where((~emptied) & centered, np.maximum(0.0, -dz), 0.0)
        lift = np.where(emptied & centered & (self.tip_pos[:, 2] < DISPENSE_CLEAR_Z), np.maximum(0.0, dz), 0.0)
        down = np.clip(-self.action[:, 2], 0.0, 1.0)
        up = np.clip(self.action[:, 2], 0.0, 1.0)
        direction = np.where(emptied, 0.25 * up - 0.50 * down, 0.25 * down - 0.10 * up)
        excess = np.maximum(0.0, down - DISPENSE_DESCENT_MAX_ACTION)
        fast = np.where((~emptied) & centered, DISPENSE_PENALTY_FAST_DESCENT * excess ** 2, 0.0)
        self.prev_curriculum_tip_z = self.tip_pos[:, 2].copy()
        self.prev_curriculum_pipette_ml = self.pipette_ml.copy()
        return {
            "curriculum_center": -20.0 * xy,
            "curriculum_aspirate": DISPENSE_REWARD_PER_ML * dispensed,
            "curriculum_vertical": DISPENSE_REWARD_VERTICAL_PER_M * (descend + lift) + direction,
            "curriculum_lateral": zeros.copy(),
            "curriculum_fast_descent": fast,
            "curriculum_success": DISPENSE_REWARD_SUCCESS * self.terms_done["success"],
        }

    def _end_to_end_reward_terms(self):
        n = self.num_envs
        zeros = np.zeros(n)

        if self.curriculum != "end_to_end":
            return {
                "curriculum_center": zeros.copy(),
                "curriculum_aspirate": zeros.copy(),
                "curriculum_vertical": zeros.copy(),
                "curriculum_lateral": zeros.copy(),
                "curriculum_fast_descent": zeros.copy(),
                "curriculum_success": zeros.copy(),
            }

        stage = self.task_stage
        goal = self._end_to_end_goal()

        # Desired Cartesian direction toward the active stage goal.
        goal_delta = goal - self.tip_pos

        desired_xy = goal_delta[:, :2]
        desired_xy_norm = np.linalg.norm(
            desired_xy,
            axis=1,
            keepdims=True,
        )

        desired_xy_unit = np.divide(
            desired_xy,
            np.maximum(desired_xy_norm, 1e-6),
        )

        action_xy = self.action[:, :2]

        # Positive when the commanded action points toward the goal,
        # negative when it points away.
        xy_alignment = np.sum(
            action_xy * desired_xy_unit,
            axis=1,
        )

        # Penalize excessive lateral action near the waypoint to prevent
        # repeated overshoot and oscillation.
        xy_distance = desired_xy_norm[:, 0]

        near_xy_goal = xy_distance <= 0.05

        near_goal_action_penalty = np.where(
            near_xy_goal,
            -0.50 * np.sum(action_xy**2, axis=1),
            0.0,
        )

        horizontal_stage = np.isin(
            stage,
            [0, 3],
        )

        horizontal_guidance = np.where(
            horizontal_stage,
            1.0 * xy_alignment + near_goal_action_penalty,
            0.0,
        )

        distance = np.linalg.norm(
            self.tip_pos - goal,
            axis=1,
        )

        distance_progress = np.clip(
            self.prev_center_above_distance - distance,
            -0.02,
            0.02,
        )

        liquid_delta = (
            self.pipette_ml
            - self.prev_curriculum_pipette_ml
        )

        aspirated = np.where(
            stage <= 2,
            np.maximum(0.0, liquid_delta),
            0.0,
        )

        dispensed = np.where(
            stage >= 4,
            np.maximum(0.0, -liquid_delta),
            0.0,
        )

        upward_action = np.clip(
            self.action[:, 2],
            0.0,
            1.0,
        )

        downward_action = np.clip(
            -self.action[:, 2],
            0.0,
            1.0,
        )

        # Stage-specific vertical direction shaping.
        vertical_reward = np.zeros(n)

        # Stage 1: descend into beaker.
        stage_1 = stage == 1
        vertical_reward[stage_1] = (
            0.25 * downward_action[stage_1]
            - 0.10 * upward_action[stage_1]
        )

        # Stage 2: lift clear of beaker.
        stage_2 = stage == 2
        vertical_reward[stage_2] = (
            0.50 * upward_action[stage_2]
            - 0.50 * downward_action[stage_2]
        )

        # Stage 4: descend into dispensing region.
        stage_4 = stage == 4
        vertical_reward[stage_4] = (
            0.25 * downward_action[stage_4]
            - 0.10 * upward_action[stage_4]
        )

        # Stage 5: lift clear of dish.
        stage_5 = stage == 5
        vertical_reward[stage_5] = (
            0.50 * upward_action[stage_5]
            - 0.50 * downward_action[stage_5]
        )

        vertical_stage = np.isin(
            stage,
            [1, 2, 4, 5],
        )

        lateral_penalty = np.where(
            vertical_stage,
            -2.0
            * np.linalg.norm(
                self.action[:, :2],
                axis=1,
            ),
            0.0,
        )

        descending_stage = np.isin(
            stage,
            [1, 4],
        )

        descent_excess = np.maximum(
            0.0,
            downward_action
            - DISPENSE_DESCENT_MAX_ACTION,
        )

        fast_descent_penalty = np.where(
            descending_stage,
            -1.0 * descent_excess**2,
            0.0,
        )

        # Avoid carrying an old distance baseline into a new stage.
        stage_bonus = (
            E2E_STAGE_BONUS
            * self.stage_advanced_this_step
        )

        center_reward = (
            80.0 * distance_progress
            + 2.0 * horizontal_guidance
            + stage_bonus
        )

        self.prev_center_above_distance = (
            distance.copy()
        )
        self.prev_curriculum_tip_z = (
            self.tip_pos[:, 2].copy()
        )
        self.prev_curriculum_pipette_ml = (
            self.pipette_ml.copy()
        )

        return {
            "curriculum_center": center_reward,
            "curriculum_aspirate": (
                30.0 * (aspirated + dispensed)
            ),
            "curriculum_vertical": vertical_reward,
            "curriculum_lateral": lateral_penalty,
            "curriculum_fast_descent": (
                fast_descent_penalty
            ),
            "curriculum_success": (
                150.0
                * self.terms_done["success"]
            ),
        }


    def _center_lift_reward_terms(self):
        """Dense shaping for centering, controlled aspiration, and vertical extraction."""
        n = self.num_envs
        zeros = np.zeros(n)
        if self.curriculum not in (
            "center_lift",
            "approach_center_lift",
        ):
            return {
                "curriculum_center": zeros.copy(),
                "curriculum_aspirate": zeros.copy(),
                "curriculum_vertical": zeros.copy(),
                "curriculum_lateral": zeros.copy(),
                "curriculum_fast_descent": zeros.copy(),
                "curriculum_success": zeros.copy(),
            }

        xy = np.linalg.norm(
            self.tip_pos[:, :2]
            - self.beaker_pos[:, :2],
            axis=1,
        )

        centered = (
            xy <= CENTER_LIFT_CENTER_RADIUS_M
        )

        enough = (
            self.pipette_ml
            >= self.target_ml - SUCCESS_TOL_ML
        )

        approaching = (
            (self.curriculum == "approach_center_lift")
            & (~self.approach_complete)
        )

        below_clear = self.tip_pos[:, 2] < CENTER_LIFT_CLEAR_Z

        center_progress = np.clip(
            self.prev_beaker_xy_dist - xy,
            -0.02,
            0.02,
        )

        approach_goal = self.beaker_pos.copy()
        approach_goal[:, 2] = APPROACH_CENTER_Z

        approach_distance = np.linalg.norm(
            self.tip_pos - approach_goal,
            axis=1,
        )

        approach_progress = np.clip(
            self.prev_center_above_distance
            - approach_distance,
            -0.02,
            0.02,
        )

        approach_proximity = np.exp(
            -25.0 * approach_distance
        )
        aspirated = np.maximum(0.0, self.pipette_ml - self.prev_curriculum_pipette_ml)
        dz = self.tip_pos[:, 2] - self.prev_curriculum_tip_z

        # Before aspiration: descending is useful only while centered.
        descend_progress = np.where((~enough) & centered & (~approaching), np.maximum(0.0, -dz), 0.0)
        # After aspiration: reward vertical lift while remaining centered.
        lift_progress = np.where(enough & centered & below_clear, np.maximum(0.0, dz), 0.0)
        vertical = CENTER_LIFT_REWARD_VERTICAL_PER_M * (descend_progress + lift_progress)

        downward_action = np.clip(
            -self.action[:, 2],
            0.0,
            1.0,
        )

        upward_action = np.clip(
            self.action[:, 2],
            0.0,
            1.0,
        )

        direction_reward = np.where(
            enough,
            0.25 * upward_action,
            0.25 * downward_action,
        )

        wrong_direction_penalty = np.where(
            enough,
            -0.50 * downward_action,
            -0.10 * upward_action,
        )

        vertical = (
            CENTER_LIFT_REWARD_VERTICAL_PER_M
            * (
                descend_progress
                + lift_progress
            )
            + direction_reward
            + wrong_direction_penalty
        )        

        lateral_mag = np.linalg.norm(self.action[:, :2], axis=1)
        lateral_penalty = np.where(
            below_clear,
            CENTER_LIFT_PENALTY_LATERAL_BELOW_RIM * lateral_mag,
            0.0,
        )
        downward_command = np.maximum(
            0.0,
            -self.action[:, 2],
        )

        descent_excess = np.maximum(
            0.0,
            downward_command - CENTER_LIFT_DESCENT_MAX_ACTION,
        )

        fast_descent = np.where(
            (~enough) & centered,
            CENTER_LIFT_PENALTY_FAST_DESCENT
            * descent_excess ** 2,
            0.0,
        )

        terms = {
            "curriculum_center": np.where(
                approaching,
                (
                    APPROACH_REWARD_PROGRESS_PER_M
                    * approach_progress
                    + APPROACH_REWARD_PROXIMITY
                    * approach_proximity
                ),
                CENTER_LIFT_REWARD_CENTER_PER_M
                * center_progress,
            ),
            "curriculum_aspirate": CENTER_LIFT_REWARD_ASPIRATE_PER_ML * aspirated,
            "curriculum_vertical": vertical,
            "curriculum_lateral": lateral_penalty,
            "curriculum_fast_descent": fast_descent,
            "curriculum_success": CENTER_LIFT_REWARD_STAGE_SUCCESS * self.terms_done["success"],
        }

        self.prev_beaker_xy_dist = xy
        self.prev_center_above_distance = (
            approach_distance
        )
        self.prev_curriculum_tip_z = self.tip_pos[:, 2].copy()
        self.prev_curriculum_pipette_ml = self.pipette_ml.copy()
        return terms

    def goal_distance(self):
        beaker_goal = self.beaker_pos.copy(); beaker_goal[:, 2] = 0.08
        if self.curriculum == "center_above":
            return np.linalg.norm(self.tip_pos - self._center_above_goal(), axis=1)
        if self.curriculum == "aspirate":
            return np.linalg.norm(self.tip_pos - self._aspirate_goal(), axis=1)
        if self.curriculum == "lift":
            return np.linalg.norm(self.tip_pos - self._lift_goal(), axis=1)
        if self.curriculum == "transit_to_dish":
            return np.linalg.norm(self.tip_pos - self._transit_goal(), axis=1)
        if self.curriculum == "dispense_lift":
            return np.linalg.norm(self.tip_pos - self._dispense_goal(), axis=1)
        if self.curriculum == "end_to_end":
            return np.linalg.norm(self.tip_pos - self._end_to_end_goal(), axis=1)
        if self.curriculum in (
            "center_lift",
            "approach_center_lift",
        ):
            lift_goal = self.beaker_pos.copy()
            lift_goal[:, 2] = CENTER_LIFT_CLEAR_Z

            aspirate_goal = self.beaker_pos.copy()
            aspirate_goal[:, 2] = ASPIRATE_GOAL_Z

            approach_goal = self.beaker_pos.copy()
            approach_goal[:, 2] = CENTER_ABOVE_Z

            enough = (
                self.pipette_ml
                >= self.target_ml - SUCCESS_TOL_ML
            )

            if self.curriculum == "approach_center_lift":
                goal = np.where(
                    (~self.approach_complete)[:, None],
                    approach_goal,
                    np.where(
                        enough[:, None],
                        lift_goal,
                        aspirate_goal,
                    ),
                )
            else:
                goal = np.where(
                    enough[:, None],
                    lift_goal,
                    aspirate_goal,
                )

            return np.linalg.norm(
                self.tip_pos - goal,
                axis=1,
            )


        dish_goal = self.dish_pos.copy(); dish_goal[:, 2] = 0.06
        to_dish = np.linalg.norm(self.tip_pos - dish_goal, axis=1)
        via_beaker = np.linalg.norm(self.tip_pos - beaker_goal, axis=1) + np.linalg.norm(beaker_goal - dish_goal, axis=1)
        return np.where(self.pipette_ml > 0.05, to_dish, via_beaker)

    def dish_at_target(self):
        fluid_ok = np.abs(self.dish_ml - self.target_ml) < SUCCESS_TOL_ML
        return fluid_ok & self._integrity_ok()

    def dish_overfilled(self): return self.dish_ml > self.target_ml + SUCCESS_TOL_ML
    def tip_below_table(self): return self.tip_pos[:, 2] < 0.0

    def _in_warmup(self):
        return self.total_env_steps < self.warmup_steps

    def _compute_terminations(self):
        self.terms_done["time_out"] = (
            self.episode_length_buf.numpy() >= self.max_episode_length
        )
        self.terms_done["success"] = self._task_success()
        self.terms_done["overfill"] = self.dish_overfilled()
        self.terms_done["tip_below_table"] = self.tip_below_table()

        if self.reward_version == "v6":
            penetration = self.penetration_event_this_step.copy()

            force_terminate_threshold = (
                CENTER_LIFT_EMERGENCY_FORCE_N
                if self.curriculum == "center_lift"
                else (LIFT_EMERGENCY_FORCE_N if self.curriculum == "lift"
                      else REWARD_CONTACT_FORCE_THRESHOLD_N)
            )
            excessive_force = (
                self.max_contact_force_this_ep >= force_terminate_threshold
            )

            if self._in_warmup():
                # During training warm-up, record and penalize unsafe behavior,
                # but do not terminate the episode immediately.
                self.terms_done["integrity_violation"] = np.zeros(
                    self.num_envs,
                    dtype=bool,
                )
                self.terms_done["excessive_contact_force"] = np.zeros(
                    self.num_envs,
                    dtype=bool,
                )
            else:
                self.terms_done["integrity_violation"] = penetration
                self.terms_done["excessive_contact_force"] = excessive_force

            self.terminated = (
                self.terms_done["success"]
                | self.terms_done["overfill"]
                | self.terms_done["tip_below_table"]
                | self.terms_done["integrity_violation"]
                | self.terms_done["excessive_contact_force"]
            )
        else:
            self.terminated = (
                self.terms_done["success"]
                | self.terms_done["overfill"]
                | self.terms_done["tip_below_table"]
            )

        self.time_outs = self.terms_done["time_out"].copy()

    def _safe_transit_bonus(self):
        carrying = self.pipette_ml > SAFE_TRANSIT_MIN_ML
        d_b = np.linalg.norm(self.tip_pos[:, :2] - self.beaker_pos[:, :2], axis=1)
        d_d = np.linalg.norm(self.tip_pos[:, :2] - self.dish_pos[:, :2], axis=1)
        in_transit = (d_b > BEAKER_RADIUS) & (d_d > DISH_RADIUS)
        above = self.tip_pos[:, 2] > SAFE_TRANSIT_MIN_Z
        moved = np.abs(self.tip_pos[:, :2] - self.prev_tip_xy).sum(axis=1) > SAFE_TRANSIT_MIN_XY_MOVE
        under_cap = self.safe_transit_this_ep < SAFE_TRANSIT_CAP_PER_EP
        eligible = carrying & in_transit & above & moved & under_cap
        bonus = np.where(eligible, REWARD_SAFE_TRANSIT, 0.0)
        self.safe_transit_this_ep += bonus
        return bonus

    def _compute_rewards(self):
        r = {
            "progress": REWARD_PROGRESS_PER_ML * self.progress,
            "success": REWARD_SUCCESS * self.terms_done["success"],
            "reach": REWARD_REACH_PER_M * self.goal_distance(),
            "time": REWARD_TIME * (~self.terminated),
            "overfill": REWARD_OVERFILL * self.dish_overfilled(),
            "action_rate": REWARD_ACTION_RATE * np.sum((self.action - self.prev_action) ** 2, axis=1),
        }
        if self.reward_version == "v6":
            r["contact_force"] = REWARD_CONTACT_FORCE_PER_N * np.maximum(0.0, self.contact_force_this_step - REWARD_CONTACT_FORCE_THRESHOLD_N)
            r["penetration_dense"] = REWARD_PENETRATION_DENSE * self.penetration_event_this_step.astype(float)
            unsafe_termination = (
                self.terms_done["integrity_violation"]
                | self.terms_done["excessive_contact_force"]
            )

            r["penetration_term"] = np.where(
                unsafe_termination,
                REWARD_PENETRATION_TERMINATE,
                0.0,
            )
            r["safe_transit"] = self._safe_transit_bonus()
            if self.curriculum == "center_above":
                r.update(self._center_above_reward_terms())
            elif self.curriculum == "aspirate":
                r.update(self._aspirate_reward_terms())
            elif self.curriculum == "lift":
                r.update(self._lift_reward_terms())
            elif self.curriculum == "transit_to_dish":
                r.update(self._transit_reward_terms())
            elif self.curriculum == "dispense_lift":
                r.update(self._dispense_reward_terms())
            elif self.curriculum == "end_to_end":
                r.update(self._end_to_end_reward_terms())
            else:
                r.update(self._center_lift_reward_terms())
        total = np.zeros(self.num_envs)
        for k, v in r.items():
            self.reward_sums[k] += v
            total += v
        return total

    def _command(self): return np.stack([self.target_ml, self.pipette_ml, self.dish_ml], axis=1)

    def _compute_obs(self):
        f32 = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=self.device)
        positions = np.concatenate([self.beaker_pos[:, :2], self.dish_pos[:, :2]], axis=1)
        approach_stage = (
            self.approach_complete
            .astype(np.float32)[:, None]
        )        
        joint_vel = np.stack([d.qvel[:9] for d in self.datas])
        stage_obs = approach_stage
        if self.curriculum == "end_to_end":
            stage_obs = np.eye(E2E_NUM_STAGES, dtype=np.float32)[self.task_stage]
        self.obs_buf = {
            "policy": f32(np.concatenate([self.tip_pos, self._command(), self.action, positions, stage_obs], axis=1)),
            "critic": f32(np.concatenate([self.tip_pos, self._command(), positions, joint_vel, self.action, stage_obs], axis=1)),
        }
        return self.obs_buf

    def _sample_offsets(self, k):
        b_nom, d_nom = np.array(BEAKER_POS[:2]), np.array(DISH_POS[:2])
        draw_b = lambda m: np.tile(np.array(self.fixed_beaker_offset), (m, 1)) if self.fixed_beaker_offset is not None else self.rng.uniform(-self.beaker_range, self.beaker_range, size=(m, 2))
        draw_d = lambda m: np.tile(np.array(self.fixed_dish_offset), (m, 1)) if self.fixed_dish_offset is not None else self.rng.uniform(-self.dish_range, self.dish_range, size=(m, 2))
        b, d = draw_b(k), draw_d(k)
        for _ in range(100):
            bad = (np.linalg.norm(b_nom + b, axis=1) < MIN_REACH_M) | (np.linalg.norm(d_nom + d, axis=1) < MIN_REACH_M)
            bad |= np.linalg.norm((b_nom + b) - (d_nom + d), axis=1) < MIN_SEPARATION_M
            if not bad.any(): break
            nb = int(bad.sum()); b[bad], d[bad] = draw_b(nb), draw_d(nb)
        else: raise RuntimeError("could not sample valid container positions")
        return b, d

    def _reset_idx(self, ids):
        if len(ids) == 0: return
        log = {}
        for k in self.reward_sums:
            log[f"Episode_Reward/{k}"] = float(self.reward_sums[k][ids].mean() / self.episode_length_s)
            self.reward_sums[k][ids] = 0.0
        for k, v in self.terms_done.items():
            log[f"Episode_Termination/{k}"] = float(np.count_nonzero(v[ids]))
        log["Metrics/fluid/dish_error_ml"] = float(np.abs(self.dish_ml[ids] - self.target_ml[ids]).mean())
        if self.reward_version == "v6":
            log["Metrics/safety/penetration_count"] = float(self.penetration_count_this_ep[ids].mean())
            log["Metrics/safety/max_contact_force_n"] = float(self.max_contact_force_this_ep[ids].mean())
            log["Metrics/safety/safe_transit_ep"] = float(self.safe_transit_this_ep[ids].mean())
            log["Metrics/safety/in_warmup"] = float(self._in_warmup())
        success = self.terms_done["success"][ids].astype(float)
        for name, off in (("dish", self.dish_offset), ("beaker", self.beaker_offset)):
            bins = distance_bin(np.linalg.norm(off[ids], axis=1))
            for b in np.unique(bins).tolist():
                log[f"Metrics/fluid/success_{name}_{distance_bin_label(b)}"] = float(success[bins == b].mean())
        self.extras["log"] = log

        if self.curriculum in (
            "aspirate",
            "lift",
            "center_lift",
            "transit_to_dish",
            "dispense_lift",
        ):
            # Begin from the validated fixed-beaker pose.
            # Keep reset deterministic for this isolated curriculum skill.
            qpos = np.tile(
                ASPIRATE_START_QPOS,
                (len(ids), 1),
            )
        else:
            noise = self.rng.uniform(
                -RESET_JOINT_NOISE,
                RESET_JOINT_NOISE,
                size=(len(ids), 9),
            )

            qpos = np.clip(
                HOME_QPOS + noise,
                self.joint_range[:, 0],
                self.joint_range[:, 1],
            )
        b_off, d_off = self._sample_offsets(len(ids))
        self.beaker_offset[ids] = b_off; self.dish_offset[ids] = d_off
        for j, i in enumerate(ids):
            d = self.datas[i]

            d.qpos[:9] = qpos[j]
            d.qvel[:] = 0.0

            d.ctrl[:7] = qpos[j, :7]
            d.ctrl[7] = 255.0

            d.mocap_pos[self.beaker_mocap] = (
                np.array(BEAKER_POS)
                + np.array([b_off[j, 0], b_off[j, 1], 0.0])
            )

            d.mocap_pos[self.dish_mocap] = (
                np.array(DISH_POS)
                + np.array([d_off[j, 0], d_off[j, 1], 0.0])
            )

            mujoco.mj_forward(self.model, d)

            if self.curriculum == "lift":
                lift_start = d.mocap_pos[
                    self.beaker_mocap
                ].copy()

                lift_start[2] = LIFT_START_Z

                self._settle_tip_target(
                    i,
                    lift_start,
                )

            elif self.curriculum == "center_lift":
                # Begin exactly centered at the validated safe height.
                center_lift_start = d.mocap_pos[
                    self.beaker_mocap
                ].copy()
                center_lift_start[2] = CENTER_ABOVE_Z
                self._settle_tip_target(
                    i,
                    center_lift_start,
                )
            elif self.curriculum == "transit_to_dish":
                transit_start = d.mocap_pos[self.beaker_mocap].copy()
                transit_start[2] = TRANSIT_Z
                self._settle_tip_target(i, transit_start)
            elif self.curriculum == "dispense_lift":
                dispense_start = d.mocap_pos[self.dish_mocap].copy()
                dispense_start[2] = DISPENSE_START_Z
                self._settle_tip_target(i, dispense_start)

            if self.curriculum in (
                "aspirate",
                "lift",
                "center_lift",
                "transit_to_dish",
                "dispense_lift",
            ):
                # Synchronize the Cartesian controller with the physical reset pose.
                self.target_pos[i] = d.site_xpos[self.tip_site]
        self.target_ml[ids] = self.rng.uniform(*TARGET_RANGE_ML, size=len(ids))
        if self.curriculum in (
            "aspirate",
            "center_lift",
            "approach_center_lift",
            "end_to_end",
        ):
            self.target_ml[ids] = ASPIRATE_TARGET_ML

        elif self.curriculum == "lift":
            self.target_ml[ids] = LIFT_TARGET_ML
        elif self.curriculum == "transit_to_dish":
            self.target_ml[ids] = TRANSIT_TARGET_ML
        elif self.curriculum == "dispense_lift":
            self.target_ml[ids] = DISPENSE_TARGET_ML

        self.pipette_ml[ids] = 0.0
        self.dish_ml[ids] = 0.0
        self.beaker_ml[ids] = BEAKER_INITIAL_ML
        if self.curriculum == "lift":
            self.pipette_ml[ids] = LIFT_TARGET_ML
            self.beaker_ml[ids] = BEAKER_INITIAL_ML - LIFT_TARGET_ML
        elif self.curriculum == "transit_to_dish":
            self.pipette_ml[ids] = TRANSIT_TARGET_ML
            self.beaker_ml[ids] = BEAKER_INITIAL_ML - TRANSIT_TARGET_ML
        elif self.curriculum == "dispense_lift":
            self.pipette_ml[ids] = DISPENSE_TARGET_ML
            self.beaker_ml[ids] = BEAKER_INITIAL_ML - DISPENSE_TARGET_ML

        self.progress[ids] = 0.0
        self.approach_complete[ids] = False
        self.task_stage[ids] = 0
        self.stage_advanced_this_step[ids] = False
        if self.curriculum not in ("aspirate", "lift", "center_lift", "transit_to_dish", "dispense_lift"):
            self.target_pos[ids] = HOME_TIP_POS
        self.action[ids] = 0.0; self.prev_action[ids] = 0.0
        self.episode_length_buf[torch.as_tensor(ids)] = 0
        self._refresh_poses(ids)
        if self.curriculum in (
            "aspirate",
            "center_lift",
        ):

            xy_error = np.linalg.norm(
                self.tip_pos[ids, :2]
                - self.beaker_pos[ids, :2],
                axis=1,
            )

            z_error = np.abs(
                self.tip_pos[ids, 2]
                - CENTER_ABOVE_Z
            )

            if np.any(xy_error > CENTER_ABOVE_XY_TOL_M):
                raise RuntimeError(
                    "Aspirate/center-lift reset is not centered "
                    "above the beaker: "
                    f"max XY error={xy_error.max():.4f} m"
                )

            if np.any(z_error > CENTER_ABOVE_Z_TOL_M):
                raise RuntimeError(
                    "Aspirate/center-lift reset has incorrect height: "
                    f"max Z error={z_error.max():.4f} m"
                )
        if self.curriculum == "lift":
            xy_error = np.linalg.norm(
                self.tip_pos[ids, :2] - self.beaker_pos[ids, :2], axis=1
            )
            z_error = np.abs(self.tip_pos[ids, 2] - LIFT_START_Z)
            if np.any(xy_error > LIFT_XY_TOL_M):
                raise RuntimeError(
                    "Lift reset is not centered inside the beaker: "
                    f"max XY error={xy_error.max():.4f} m"
                )
            if np.any(z_error > 0.02):
                raise RuntimeError(
                    "Lift reset has incorrect depth: "
                    f"max Z error={z_error.max():.4f} m"
                )

        center_above_goal = self.beaker_pos[ids].copy()
        center_above_goal[:, 2] = CENTER_ABOVE_Z
        curriculum_goal = center_above_goal.copy()
        if self.curriculum == "aspirate":
            curriculum_goal[:, 2] = ASPIRATE_GOAL_Z
        elif self.curriculum == "lift":
            curriculum_goal[:, 2] = LIFT_CLEAR_Z
        elif self.curriculum == "transit_to_dish":
            curriculum_goal = self.dish_pos[ids].copy()
            curriculum_goal[:, 2] = TRANSIT_Z
        elif self.curriculum == "dispense_lift":
            curriculum_goal = self.dish_pos[ids].copy()
            curriculum_goal[:, 2] = DISPENSE_GOAL_Z
        self.prev_center_above_distance[ids] = np.linalg.norm(
            self.tip_pos[ids] - curriculum_goal, axis=1
        )
        self.prev_beaker_xy_dist[ids] = np.linalg.norm(
            self.tip_pos[ids, :2] - self.beaker_pos[ids, :2], axis=1
        )
        self.prev_curriculum_tip_z[ids] = self.tip_pos[ids, 2]
        self.prev_curriculum_pipette_ml[ids] = self.pipette_ml[ids]
        self.contact_force_this_step[ids] = 0.0
        self.penetration_event_this_step[ids] = False
        self.penetration_count_this_ep[ids] = 0
        self.max_contact_force_this_ep[ids] = 0.0
        self.safe_transit_this_ep[ids] = 0.0
        self.prev_dist_beaker[ids] = np.linalg.norm(self.tip_pos[ids, :2] - self.beaker_pos[ids, :2], axis=1)
        self.prev_tip_z[ids] = self.tip_pos[ids, 2]
        self.prev_tip_xy[ids] = self.tip_pos[ids, :2]

        if self.curriculum == "center_lift":
            xy_error = np.linalg.norm(
                self.tip_pos[ids, :2]
                - self.beaker_pos[ids, :2],
                axis=1,
            )

            log["Metrics/curriculum/beaker_xy_error_m"] = float(
                xy_error.mean()
            )

            log["Metrics/curriculum/pipette_ml"] = float(
                self.pipette_ml[ids].mean()
            )

            log["Metrics/curriculum/tip_z_m"] = float(
                self.tip_pos[ids, 2].mean()
            )        

    def reset(self):
        self._reset_idx(np.arange(self.num_envs))
        self.extras = {}
        return self._compute_obs(), self.extras

    def step(self, actions):
        a = actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions)
        self.prev_action[:] = self.action
        self.action[:] = np.clip(a, -1.0, 1.0)

        if self.curriculum in (
            "aspirate",
            "lift",
            "center_lift",
            "dispense_lift",
        ):
            # These isolated curricula are vertical-only.
            self.action[:, 0] = 0.0
            self.action[:, 1] = 0.0

        if self.curriculum == "transit_to_dish":
            # Transit is horizontal-only at a fixed safe clearance height.
            self.action[:, 2] = 0.0
            self.target_pos[:, 2] = TRANSIT_Z
        if self.curriculum == "end_to_end":
            stage = self.task_stage

            bxy = np.linalg.norm(
                self.tip_pos[:, :2]
                - self.beaker_pos[:, :2],
                axis=1,
            )
            dxy = np.linalg.norm(
                self.tip_pos[:, :2]
                - self.dish_pos[:, :2],
                axis=1,
            )

            # Stages 1, 2, 4, and 5 are vertical-only.
            vertical = np.isin(
                stage,
                [1, 2, 4, 5],
            )
            self.action[vertical, 0] = 0.0
            self.action[vertical, 1] = 0.0

            # Stage 0: goal-directed horizontal approach to the beaker.
            stage_0 = stage == 0

            beaker_delta_xy = (
                self.beaker_pos[:, :2]
                - self.tip_pos[:, :2]
            )

            beaker_xy_distance = np.linalg.norm(
                beaker_delta_xy,
                axis=1,
            )

            beaker_guidance = np.clip(
                beaker_delta_xy / STEP_SIZE,
                -0.35,
                0.35,
            )

            near_beaker = (
                stage_0
                & (beaker_xy_distance <= 0.05)
            )

            beaker_guidance_near = np.clip(
                beaker_delta_xy / STEP_SIZE,
                -0.15,
                0.15,
            )

            # Override PPO lateral commands with safe goal-directed guidance.
            self.action[stage_0, 0] = (
                beaker_guidance[stage_0, 0]
            )
            self.action[stage_0, 1] = (
                beaker_guidance[stage_0, 1]
            )

            # Slow the approach inside 5 cm.
            self.action[near_beaker, 0] = (
                beaker_guidance_near[near_beaker, 0]
            )
            self.action[near_beaker, 1] = (
                beaker_guidance_near[near_beaker, 1]
            )

            # Stage 0 is horizontal-only. Never let an uncentered
            # approach command the tip upward or downward.
            self.action[stage_0, 2] = 0.0

            # Stage 1: goal-directed controlled descent into the beaker.
            stage_1 = stage == 1

            beaker_descent_error = (
                ASPIRATE_GOAL_Z
                - self.tip_pos[:, 2]
            )

            beaker_descent_guidance = np.clip(
                beaker_descent_error / STEP_SIZE,
                -ASPIRATE_DESCENT_MAX_ACTION,
                0.0,
            )

            # Override the fixed-position PPO command so descent cannot stall.
            self.action[stage_1, 2] = (
                beaker_descent_guidance[stage_1]
            )


            # Stage 2: goal-directed vertical extraction from the beaker.
            stage_2 = stage == 2

            beaker_lift_error = (
                CENTER_LIFT_CLEAR_Z
                - self.tip_pos[:, 2]
            )

            beaker_lift_guidance = np.clip(
                beaker_lift_error / STEP_SIZE,
                0.0,
                0.35,
            )

            # Override the fixed-position PPO command so extraction cannot stall.
            self.action[stage_2, 2] = (
                beaker_lift_guidance[stage_2]
            )

            # Stage 3: goal-directed horizontal transit.
            stage_3 = stage == 3

            dish_delta_xy = (
                self.dish_pos[:, :2]
                - self.tip_pos[:, :2]
            )

            dish_xy_distance = np.linalg.norm(
                dish_delta_xy,
                axis=1,
            )

            dish_guidance = np.clip(
                dish_delta_xy / STEP_SIZE,
                -0.35,
                0.35,
            )

            near_dish = (
                stage_3
                & (dish_xy_distance <= 0.05)
            )

            dish_guidance_near = np.clip(
                dish_delta_xy / STEP_SIZE,
                -0.15,
                0.15,
            )

            self.action[stage_3, 0] = (
                dish_guidance[stage_3, 0]
            )
            self.action[stage_3, 1] = (
                dish_guidance[stage_3, 1]
            )

            self.action[near_dish, 0] = (
                dish_guidance_near[near_dish, 0]
            )
            self.action[near_dish, 1] = (
                dish_guidance_near[near_dish, 1]
            )

            self.action[stage_3, 2] = 0.0
            self.target_pos[stage_3, 2] = TRANSIT_Z

            # Stage 4: goal-directed descent into the dish.
            stage_4 = stage == 4

            dish_descent_error = (
                DISPENSE_GOAL_Z
                - self.tip_pos[:, 2]
            )

            dish_descent_guidance = np.clip(
                dish_descent_error / STEP_SIZE,
                -DISPENSE_DESCENT_MAX_ACTION,
                0.0,
            )

            self.action[stage_4, 2] = (
                dish_descent_guidance[stage_4]
            )

            # Stage 5: goal-directed extraction from the dish.
            stage_5 = stage == 5

            dish_lift_error = (
                DISPENSE_CLEAR_Z
                - self.tip_pos[:, 2]
            )

            dish_lift_guidance = np.clip(
                dish_lift_error / STEP_SIZE,
                0.0,
                0.35,
            )

            self.action[stage_5, 2] = (
                dish_lift_guidance[stage_5]
            )


        if self.curriculum == "approach_center_lift":
            xy_error = np.linalg.norm(
                self.tip_pos[:, :2] - self.beaker_pos[:, :2], axis=1
            )
            z_error = np.abs(self.tip_pos[:, 2] - APPROACH_CENTER_Z)
            reached_waypoint = (
                (xy_error <= APPROACH_CENTER_XY_TOL_M)
                & (z_error <= APPROACH_CENTER_Z_TOL_M)
            )
            # One-way transition: lateral control never reopens this episode.
            self.approach_complete |= reached_waypoint
            before_waypoint = ~self.approach_complete
            off_center_before_waypoint = (
                before_waypoint
                & (
                    xy_error
                    > APPROACH_CENTER_XY_TOL_M
                )
            )

            # Block descent only while horizontally off-center.
            # Once XY is centered, permit vertical descent to the
            # center-above waypoint so the latch can activate.
            self.action[
                off_center_before_waypoint,
                2,
            ] = np.maximum(
                self.action[
                    off_center_before_waypoint,
                    2,
                ],
                0.0,
            )
            # Descent, aspiration, and extraction are vertical-only.
            self.action[self.approach_complete, 0] = 0.0
            self.action[self.approach_complete, 1] = 0.0
        self.target_pos += self.action * STEP_SIZE
        np.clip(self.target_pos, self.ws_min, self.ws_max, out=self.target_pos)
        self._apply_actions()

        if self.viewer is not None:
            self.viewer.sync()

        self.episode_length_buf += 1
        self.total_env_steps += self.num_envs

        # Update physical state and safety measurements first.
        self._update_safety()

        # Update aspiration/dispensing before success, termination,
        # reward, logging, or automatic reset is evaluated.
        self._update_fluid()

        # Advance the end-to-end state only after current poses and fluid are known.
        self._advance_end_to_end_stage()
        # Termination and reward now observe the current fluid state.
        self._compute_terminations()

        # Capture timeout state before automatic reset destroys the episode state.
        if args_cli.diagnose_timeouts:
            timeout_ids = np.flatnonzero(
                self.terms_done["time_out"]
                & ~self.terms_done["success"]
            )

            for i in timeout_ids.tolist():
                bxy = float(
                    np.linalg.norm(
                        self.tip_pos[i, :2]
                        - self.beaker_pos[i, :2]
                    )
                )
                dxy = float(
                    np.linalg.norm(
                        self.tip_pos[i, :2]
                        - self.dish_pos[i, :2]
                    )
                )

                active_goal = self._end_to_end_goal()[i]
                goal_error = float(
                    np.linalg.norm(
                        self.tip_pos[i]
                        - active_goal
                    )
                )

                print(
                    "[Timeout] "
                    f"env={i} "
                    f"episode_step={int(self.episode_length_buf[i])} "
                    f"stage={int(self.task_stage[i])} "
                    f"tip={self.tip_pos[i].round(4).tolist()} "
                    f"target_pos={self.target_pos[i].round(4).tolist()} "
                    f"active_goal={active_goal.round(4).tolist()} "
                    f"goal_error_m={goal_error:.4f} "
                    f"beaker_offset={self.beaker_offset[i].round(4).tolist()} "
                    f"dish_offset={self.dish_offset[i].round(4).tolist()} "
                    f"beaker_xy_error_m={bxy:.4f} "
                    f"dish_xy_error_m={dxy:.4f} "
                    f"target_ml={self.target_ml[i]:.2f} "
                    f"beaker_ml={self.beaker_ml[i]:.2f} "
                    f"pipette_ml={self.pipette_ml[i]:.2f} "
                    f"dish_ml={self.dish_ml[i]:.2f} "
                    f"action={self.action[i].round(4).tolist()} "
                    f"max_contact_force_n="
                    f"{self.max_contact_force_this_ep[i]:.3f} "
                    f"penetration_count="
                    f"{int(self.penetration_count_this_ep[i])} "
                    f"integrity_ok={bool(self._integrity_ok()[i])}"
                )

        reward = self._compute_rewards()
        self.extras = {}

        if self.auto_reset:
            self._reset_idx(
                np.nonzero(
                    self.terminated | self.time_outs
                )[0]
            )

        self.prev_tip_xy[:] = self.tip_pos[:, :2]
        obs = self._compute_obs()
        t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=self.device)
        return obs, t(reward, torch.float32), t(self.terminated, torch.bool), t(self.time_outs, torch.bool), self.extras

    def pipette_tilt_deg(self):
        z = np.stack([d.xmat[self.hand_id].reshape(3, 3)[:, 2] for d in self.datas])
        return np.degrees(np.arccos(np.clip(-z[:, 2], -1.0, 1.0)))

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
        self.cfg = {"task": "liquid_transfer_m3_mujoco", "num_envs": env.num_envs, "episode_length_s": env.episode_length_s}

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
    mlp = lambda dist: {"class_name": "MLPModel", "hidden_dims": [256, 128], "activation": "elu",
                        "obs_normalization": True, "distribution_cfg": dist}
    return {
        "seed": seed, "device": device, "num_steps_per_env": 32,
        "init_at_random_ep_len": True, "max_iterations": max_iterations,
        "empirical_normalization": {},
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "clip_actions": 1.0, "check_for_nan": True, "save_interval": 25,
        "experiment_name": "liquid_transfer_m3", "run_name": run_name,
        "logger": "tensorboard", "resume": False, "load_run": ".*",
        "load_checkpoint": "model_.*.pt", "class_name": "OnPolicyRunner",
        "actor": mlp({"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"}),
        "critic": mlp(None),
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 5, "num_mini_batches": 4,
                      "learning_rate": 3.0e-4, "schedule": "adaptive", "gamma": 0.99, "lam": 0.95,
                      "entropy_coef": 0.003, "desired_kl": 0.01, "max_grad_norm": 1.0,
                      "optimizer": "adam", "value_loss_coef": 1.0, "use_clipped_value_loss": True,
                      "clip_param": 0.2, "normalize_advantage_per_mini_batch": False,
                      "share_cnn_encoders": False, "use_mixed_precision": False,
                      "rnd_cfg": None, "symmetry_cfg": None},
        "torch_compile_mode": None, "policy": {},
    }


def _pace(t_last):
    if args_cli.viz: time.sleep(max(0.0, t_last + 1.0 / CONTROL_HZ - time.perf_counter()))
    return time.perf_counter()

def _count_terminations(env, counts):
    for name, arr in env.terms_done.items():
        k = int(arr.sum())
        if k: counts[name] = counts.get(name, 0) + k

def run_random(env, max_steps):
    obs, _ = env.reset()
    print("[Obs] " + ", ".join(f"{k}: {tuple(v.shape)}" for k, v in obs.items()))
    t = t0 = time.perf_counter(); done_counts = {}
    for step in range(max_steps):
        if not env.viewer_running(): break
        _, rew, _, _, _ = env.step(env.rng.uniform(-1.0, 1.0, size=(env.num_envs, 3)))
        _count_terminations(env, done_counts)
        if step % 100 == 0:
            print(f"[step {step:4d}] tip0={env.tip_pos[0].round(3).tolist()} reward0={rew[0]:.3f} pen0={env.penetration_count_this_ep[0]}")
        t = _pace(t)
    print(f"[Random] {time.perf_counter()-t0:.1f}s; terminations: {done_counts}")

def run_probe(env):
    env.reset(); env.auto_reset = False
    env.step(np.zeros((env.num_envs, 3)))
    for name, goal_z, get_pos in (("beaker fill", 0.08, lambda: env.beaker_pos), ("dish dispense", 0.06, lambda: env.dish_pos)):
        for _ in range(150):
            goal = get_pos().copy(); goal[:, 2] = goal_z
            env.step(np.clip((goal - env.target_pos) / STEP_SIZE, -1.0, 1.0))
        goal = get_pos().copy(); goal[:, 2] = goal_z
        err = np.linalg.norm(env.tip_pos - goal, axis=1)
        for i in range(env.num_envs):
            print(f"[Probe] env {i} -> {name}: error {100*err[i]:.1f}cm")

def run_scripted(env, max_steps):
    env.reset(); n = env.num_envs
    #if args_cli.curriculum == "end_to_end":
    #    env.auto_reset = False    
    if args_cli.curriculum in (
        "aspirate",
        "center_lift",
    ):
        # Start centered above the beaker and descend.
        phase = np.ones(n, dtype=int)

    elif args_cli.curriculum == "lift":
        # Start inside with 2 mL and lift.
        phase = np.full(n, 2, dtype=int)
    elif args_cli.curriculum == "transit_to_dish":
        # Start lifted above the beaker and move to the high dish waypoint.
        phase = np.full(n, 3, dtype=int)
    elif args_cli.curriculum == "dispense_lift":
        phase = np.full(n, 4, dtype=int)
    else:
        phase = np.zeros(n, dtype=int)

    done_counts = {}; ep_lens = []; ep_len = np.zeros(n)
    t = t0 = time.perf_counter()
    for step in range(max_steps):
        if not env.viewer_running(): break
        b_g, d_g = env.beaker_pos.copy(), env.dish_pos.copy()
        hb, hd = b_g.copy(), d_g.copy(); hb[:, 2] = 0.25; b_g[:, 2] = 0.08; hd[:, 2] = 0.20; d_g[:, 2] = 0.06
        if args_cli.curriculum == "end_to_end":
            goal = np.stack(
                [hb, b_g, hb, hd, d_g, hd],
                axis=1,
            )[np.arange(n), env.task_stage]

            # Match the actual validated clearance heights.
            goal[env.task_stage == 0, 2] = CENTER_LIFT_CLEAR_Z
            goal[env.task_stage == 2, 2] = CENTER_LIFT_CLEAR_Z
            goal[env.task_stage == 3, 2] = TRANSIT_Z
            goal[env.task_stage == 5, 2] = DISPENSE_CLEAR_Z

        else:
            goal = np.stack(
                [hb, b_g, hb, hd, d_g, hd],
                axis=1,
            )[np.arange(n), phase]
        _, rew, terminated, truncated, _ = env.step(np.clip((goal - env.target_pos) / STEP_SIZE, -1.0, 1.0))
        if step % 25 == 0:
            unique, counts = np.unique(
                env.task_stage,
                return_counts=True,
            )
            stages = dict(zip(unique.tolist(), counts.tolist()))
            print(
                f"[E2E step {step:4d}] "
                f"stages={stages} "
                f"tip0={env.tip_pos[0].round(3).tolist()} "
                f"pipette0={env.pipette_ml[0]:.2f} "
                f"dish0={env.dish_ml[0]:.2f} "
                f"phase0={phase[0]}"
            )        
        ep_len += 1

        if args_cli.curriculum == "end_to_end":
            done_now = terminated | truncated

            if done_now.any():
                print(
                    "[E2E final] "
                    f"stage={env.task_stage[0]} "
                    f"target={env.target_ml[0]:.2f} "
                    f"pipette={env.pipette_ml[0]:.2f} "
                    f"dish={env.dish_ml[0]:.2f} "
                    f"tip_z={env.tip_pos[0, 2]:.3f} "
                    f"pen={env.penetration_count_this_ep[0]} "
                    f"max_cf={env.max_contact_force_this_ep[0]:.1f}N "
                    f"success={env.terms_done['success'][0]}"
                )
                _count_terminations(env, done_counts)
                ep_lens += ep_len[done_now.cpu().numpy()].tolist()
                break

        if args_cli.curriculum == "end_to_end":
            # The environment owns the monotonic six-stage state machine.
            phase[:] = env.task_stage
        else:
            reached = np.linalg.norm(
                env.tip_pos - goal,
                axis=1,
            ) < 0.015

            need = np.minimum(
                env.target_ml - env.dish_ml,
                PIPETTE_CAPACITY_ML,
            )

            reached &= (
                (phase != 1)
                | (env.pipette_ml >= need - 1e-3)
            )

            reached &= (
                (phase != 4)
                | (
                    env.dish_ml
                    >= env.target_ml - SUCCESS_TOL_ML
                )
            )

            phase = np.where(
                reached,
                (phase + 1) % 6,
                phase,
            )
            dones = (terminated | truncated).cpu().numpy()
            
        
            if dones.any():
                if args_cli.curriculum in (
                    "aspirate",
                    "center_lift",
                ):
                    phase[dones] = 1

                elif args_cli.curriculum == "lift":
                    phase[dones] = 2
                elif args_cli.curriculum == "transit_to_dish":
                    phase[dones] = 3
                elif args_cli.curriculum == "dispense_lift":
                    phase[dones] = 4
                else:
                    phase[dones] = 0

                ep_lens += ep_len[dones].tolist()
                ep_len[dones] = 0
            _count_terminations(env, done_counts)
        t = _pace(t)
    print(f"[Scripted] terminations: {done_counts}; mean episode len: {sum(ep_lens)/max(len(ep_lens),1):.0f}; {time.perf_counter()-t0:.1f}s")

def run_train(env, agent_cfg, log_dir, resume_path):
    from rsl_rl.runners import OnPolicyRunner
    runner = OnPolicyRunner(RslRlVecEnv(env, agent_cfg["clip_actions"]), agent_cfg, log_dir=log_dir, device=agent_cfg["device"])
    if resume_path: runner.load(resume_path, map_location=agent_cfg["device"])
    print(f"[Train] reward_version={REWARD_VERSION}, warmup_steps={env.warmup_steps}, logging to {log_dir}")
    runner.learn(num_learning_iterations=agent_cfg["max_iterations"], init_at_random_ep_len=True)

def run_play(env, agent_cfg, checkpoint, max_steps, hold=False):
    from rsl_rl.runners import OnPolicyRunner
    venv = RslRlVecEnv(env, agent_cfg["clip_actions"])
    runner = OnPolicyRunner(venv, agent_cfg, log_dir=None, device=agent_cfg["device"])
    runner.load(checkpoint, map_location=agent_cfg["device"])
    policy = runner.get_inference_policy(device=env.device)
    obs = venv.get_observations(); n = env.num_envs
    counts = {}; per_bin = {"dish": {}, "beaker": {}}; ep_lens = []; ep_len = np.zeros(n)
    hold_limit = int(round(args_cli.episode_length_s * CONTROL_HZ))
    outcome = None; t = t0 = time.perf_counter()
    with torch.inference_mode():
        for step in range(max_steps):
            if not env.viewer_running(): break
            dish_before, beaker_before = env.dish_offset.copy(), env.beaker_offset.copy()
            obs, _, dones, _ = venv.step(policy(obs))
            ep_len += 1; _count_terminations(env, counts)
            done_ids = np.nonzero(dones.cpu().numpy())[0]
            if len(done_ids) and not hold:
                succ = env.terms_done["success"]
                for name, off in (("dish", dish_before), ("beaker", beaker_before)):
                    bins = distance_bin(np.linalg.norm(off, axis=1))
                    for i in done_ids.tolist():
                        st = per_bin[name].setdefault(int(bins[i]), [0, 0])
                        st[0] += int(succ[i]); st[1] += 1
                ep_lens += ep_len[done_ids].tolist(); ep_len[done_ids] = 0
            if step % 50 == 0:
                print(f"[step {step:4d}] dish0={env.dish_ml[0]:.2f} pen0={env.penetration_count_this_ep[0]}")
            if hold:
                if (
                    REWARD_VERSION == "v6"
                    and env.contact_force_this_step[0]
                    >= REWARD_CONTACT_FORCE_THRESHOLD_N
                ):
                    outcome = "excessive_contact_force"
                elif (
                    REWARD_VERSION == "v6"
                    and env.penetration_event_this_step[0]
                ):
                    outcome = "penetration_event"
                elif env.dish_at_target()[0]:
                    outcome = "success"
                elif env.dish_overfilled()[0]:
                    outcome = "overfill"
                elif env.tip_below_table()[0]:
                    outcome = "tip_below_table"
                elif step + 1 >= hold_limit:
                    outcome = "time_out"
                if outcome:
                    print(f"[Hold] ended at step {step+1}: {outcome}; beaker off {env.beaker_offset[0].round(3).tolist()} "
                          f"dish off {env.dish_offset[0].round(3).tolist()}; target={env.target_ml[0]:.2f} dish={env.dish_ml[0]:.2f} "
                          f"pen_count={env.penetration_count_this_ep[0]} max_cf={env.max_contact_force_this_ep[0]:.1f}N")
                    break
            t = _pace(t)
    if hold:
        if outcome is None: print(f"[Hold] max_steps {max_steps} reached before episode ended")
        if env.viewer is None: print("[Hold] no window"); return
        print("[Hold] final frame; close window to exit")
        while env.viewer.is_running(): env.viewer.sync(); time.sleep(0.05)
        return
    episodes = sum(counts.values())
    rate = counts.get("success", 0) / max(episodes, 1)
    mean_len = sum(ep_lens) / max(len(ep_lens), 1)
    print(f"[Play] reward_version={REWARD_VERSION}; {max_steps} steps x {n} envs: {episodes} eps, outcomes {counts}, "
          f"success {rate:.1%}, mean ep len {mean_len:.0f}; {time.perf_counter()-t0:.1f}s")
    for name in ("dish", "beaker"):
        for b in sorted(per_bin[name]):
            s, k = per_bin[name][b]
            print(f"[Play] {name} {distance_bin_label(b)}: {s}/{k} = {s/max(k,1):.1%}")


def _load_policy_from_checkpoint(checkpoint_path, device="cpu"):
    """Load actor MLP + obs normalizer from an rsl_rl checkpoint. Returns a policy(obs)->actions callable."""
    import torch.nn as nn
    ckpt = torch.load(checkpoint_path, map_location=device)
    actor_sd = ckpt.get("actor_state_dict", ckpt)

    mlp_keys = sorted([k for k in actor_sd if k.startswith("mlp.") and k.endswith(".weight")],
                      key=lambda s: int(s.split(".")[1]))
    if not mlp_keys:
        print("Checkpoint keys:", list(actor_sd.keys()))
        raise RuntimeError("no mlp.*.weight keys found in checkpoint")

    in_dim = actor_sd[mlp_keys[0]].shape[1]

    layers = []
    prev = in_dim
    for i, k in enumerate(mlp_keys):
        out_dim = actor_sd[k].shape[0]
        layers.append(nn.Linear(prev, out_dim))
        prev = out_dim
        if i < len(mlp_keys) - 1:
            layers.append(nn.ELU())
    net = nn.Sequential(*layers).to(device)

    mlp_state = {k.split(".", 1)[1]: v for k, v in actor_sd.items() if k.startswith("mlp.")}
    net.load_state_dict(mlp_state)
    net.eval()

    if "obs_normalizer._mean" in actor_sd:
        obs_mean = actor_sd["obs_normalizer._mean"].to(device)
        obs_std = actor_sd["obs_normalizer._std"].to(device)
    else:
        obs_mean = torch.zeros(in_dim, device=device)
        obs_std = torch.ones(in_dim, device=device)

    print(f"[Policy] loaded {checkpoint_path}: in_dim={in_dim}, layers={[l for l in layers if isinstance(l, nn.Linear)].__len__()}, "
          f"norm={'yes' if 'obs_normalizer._mean' in actor_sd else 'no'}")

    def policy(obs):
        with torch.no_grad():
            return net((obs - obs_mean) / (obs_std + 1e-8))

    return policy


def run_play_manual(env, checkpoint_path, max_steps, hold=False):
    """Play mode without rsl_rl: torch.load the checkpoint, run the actor directly."""
    device = env.device
    recorder = None
    video_writer = None
    video_path = None
    if args_cli.record_mp4:
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise SystemExit(
                "MP4 recording needs imageio and imageio-ffmpeg. Install with: "
                "python -m pip install imageio imageio-ffmpeg"
            ) from exc
        if args_cli.record_width <= 0 or args_cli.record_height <= 0:
            raise SystemExit("--record_width and --record_height must be positive")
        if args_cli.record_fps <= 0:
            raise SystemExit("--record_fps must be positive")
        video_path = os.path.abspath(args_cli.record_mp4)
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        recorder = mujoco.Renderer(
            env.model,
            height=args_cli.record_height,
            width=args_cli.record_width,
        )
        try:
            camera_id = env.model.camera(args_cli.record_camera).id
        except KeyError as exc:
            recorder.close()
            raise SystemExit(
                f"Unknown recording camera: {args_cli.record_camera!r}"
            ) from exc
        video_writer = imageio.get_writer(
            video_path,
            fps=args_cli.record_fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        print(
            f"[Record] MP4={video_path}; "
            f"size={args_cli.record_width}x{args_cli.record_height}; "
            f"fps={args_cli.record_fps}; camera={args_cli.record_camera}"
        )
    policy = _load_policy_from_checkpoint(checkpoint_path, device)
    obs_dict, _ = env.reset()
    n = env.num_envs
    counts = {}
    per_bin = {"dish": {}, "beaker": {}}
    ep_lens = []
    ep_len = np.zeros(n)
    hold_limit = int(round(args_cli.episode_length_s * CONTROL_HZ))
    outcome = None
    t = t0 = time.perf_counter()
    for step in range(max_steps):
        if not env.viewer_running():
            break
        dish_before, beaker_before = env.dish_offset.copy(), env.beaker_offset.copy()
        actions = policy(obs_dict["policy"])
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        if recorder is not None:
            recorder.update_scene(env.datas[0], camera=camera_id)
            video_writer.append_data(recorder.render())
        ep_len += 1
        _count_terminations(env, counts)
        done_ids = np.nonzero((terminated | truncated).cpu().numpy())[0]
        if len(done_ids) and not hold:
            succ = env.terms_done["success"]
            for name, off in (("dish", dish_before), ("beaker", beaker_before)):
                bins = distance_bin(np.linalg.norm(off, axis=1))
                for i in done_ids.tolist():
                    st = per_bin[name].setdefault(int(bins[i]), [0, 0])
                    st[0] += int(succ[i])
                    st[1] += 1
            ep_lens += ep_len[done_ids].tolist()
            ep_len[done_ids] = 0
        if step % 25 == 0:
            print(
                f"[step {step:4d}] "
                f"stage0={env.task_stage[0]} "
                f"tip0={env.tip_pos[0].round(3).tolist()} "
                f"pipette0={env.pipette_ml[0]:.2f} "
                f"dish0={env.dish_ml[0]:.2f} "
                f"action0={env.action[0].round(3).tolist()} "
                f"step_cf0={env.contact_force_this_step[0]:.1f}N "
                f"max_cf0={env.max_contact_force_this_ep[0]:.1f}N "
                f"pen0={env.penetration_count_this_ep[0]}"
            )
        if hold:
            # Use the termination decision already computed by env.step().
            # Safety outcomes take precedence over task outcomes.
            if (
                REWARD_VERSION == "v6"
                and env.terms_done["excessive_contact_force"][0]
            ):
                outcome = "excessive_contact_force"

            elif (
                REWARD_VERSION == "v6"
                and env.terms_done["integrity_violation"][0]
            ):
                outcome = "penetration_event"

            elif env.terms_done["success"][0]:
                outcome = "success"

            elif env.terms_done["overfill"][0]:
                outcome = "overfill"

            elif env.terms_done["tip_below_table"][0]:
                outcome = "tip_below_table"

            elif env.terms_done["time_out"][0]:
                outcome = "time_out"

            if outcome:
                print(
                    f"[Hold] ended at step {step+1}: {outcome}; "
                    f"target={env.target_ml[0]:.2f} "
                    f"pipette={env.pipette_ml[0]:.2f} "
                    f"dish={env.dish_ml[0]:.2f} "
                    f"tip_z={env.tip_pos[0, 2]:.3f} "
                    f"pen={env.penetration_count_this_ep[0]} "
                    f"step_cf={env.contact_force_this_step[0]:.1f}N "
                    f"max_cf={env.max_contact_force_this_ep[0]:.1f}N "
                    f"terminated={bool(terminated[0])}"
                )
                break
        if recorder is not None and bool((terminated | truncated)[0]):
            outcome = next(
                (
                    name
                    for name in (
                        "excessive_contact_force",
                        "integrity_violation",
                        "success",
                        "overfill",
                        "tip_below_table",
                        "time_out",
                    )
                    if name in env.terms_done and env.terms_done[name][0]
                ),
                "done",
            )
            print(f"[Record] ended at step {step + 1}: {outcome}")
            break
        t = _pace(t)
    if video_writer is not None:
        video_writer.close()
        recorder.close()
        print(f"[Record] saved {video_path}")
    if hold:
        if outcome is None:
            print(f"[Hold] max_steps {max_steps} reached")
        if env.viewer is None:
            return
        print("[Hold] final frame; close window to exit")
        while env.viewer.is_running():
            env.viewer.sync()
            time.sleep(0.05)
        return
    episodes = sum(counts.values())
    rate = counts.get("success", 0) / max(episodes, 1)
    mean_len = sum(ep_lens) / max(len(ep_lens), 1)
    print(f"[Play] reward_version={REWARD_VERSION}; {max_steps} steps x {n} envs: {episodes} eps, outcomes {counts}, "
          f"success {rate:.1%}, mean ep len {mean_len:.0f}; {time.perf_counter()-t0:.1f}s")
    for name in ("dish", "beaker"):
        for b in sorted(per_bin[name]):
            s, k = per_bin[name][b]
            print(f"[Play] {name} {distance_bin_label(b)}: {s}/{k} = {s/max(k,1):.1%}")

def _parse_offset(s, what):
    if not s: return None
    off = tuple(float(v) for v in s.split(",")); print(f"[{what}] fixed offset {off}"); return off

def main():
    device = args_cli.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[Device] CUDA unavailable, using cpu"); device = "cpu"
    num_steps_per_env = 32  # must match make_ppo_cfg
    total_training_steps = args_cli.max_iterations * num_steps_per_env * args_cli.num_envs
    env = LiquidTransferM3Env(
        num_envs=args_cli.num_envs, seed=args_cli.seed,
        episode_length_s=1.0e6 if args_cli.hold else args_cli.episode_length_s,
        dish_range=args_cli.dish_range, beaker_range=args_cli.beaker_range,
        fixed_dish_offset=_parse_offset(args_cli.dish_offset, "Dish"),
        fixed_beaker_offset=_parse_offset(args_cli.beaker_offset, "Beaker"),
        device=device, threads=args_cli.threads,
        auto_reset=not (args_cli.hold or bool(args_cli.record_mp4)),
        total_training_steps=total_training_steps,
        reward_version=REWARD_VERSION,
        warmup_frac=WARMUP_FRAC,
        curriculum=args_cli.curriculum,
    )
    if args_cli.viz: env.launch_viewer()
    print(f"*** model 3 v6 (MuJoCo {mujoco.__version__}): {args_cli.num_envs} envs, mode={args_cli.mode}, "
          f"reward={REWARD_VERSION}, curriculum={args_cli.curriculum}, warmup_steps={env.warmup_steps}, torch device {device} ***")
    try:
        if args_cli.mode == "random": run_random(env, args_cli.max_steps)
        elif args_cli.mode == "probe": run_probe(env)
        elif args_cli.mode == "scripted": run_scripted(env, args_cli.max_steps)
        else:
            torch.manual_seed(args_cli.seed)
            cfg = make_ppo_cfg(args_cli.seed, device, args_cli.max_iterations, args_cli.run_name)
            if args_cli.mode == "train":
                name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{args_cli.run_name}" if args_cli.run_name else "")
                run_train(env, cfg, os.path.join(HERE, "logs", "rsl_rl", cfg["experiment_name"], name), args_cli.checkpoint)
            else:
                if not args_cli.checkpoint: raise SystemExit("play needs --checkpoint <.pt>")
                run_play_manual(env, args_cli.checkpoint, args_cli.max_steps, hold=args_cli.hold)
    finally:
        env.close()

if __name__ == "__main__":
    main()
