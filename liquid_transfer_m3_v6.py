liquid_transfer_m3_v6.py

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

from __future__ import annotations

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
parser.add_argument("--viz", action="store_true")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--threads", type=int, default=0)
parser.add_argument("--reward_version", choices=["v4", "v6"], default="v6")
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

REWARD_TERMS_V4 = ("progress", "success", "reach", "time", "overfill", "action_rate")
REWARD_TERMS_V6 = REWARD_TERMS_V4 + ("contact_force", "penetration_dense", "penetration_term", "safe_transit")
TERM_V4 = ("time_out", "success", "overfill", "tip_below_table")
TERM_V6 = TERM_V4 + ("integrity_violation",)


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
                 total_training_steps=0, reward_version="v6", warmup_frac=0.20):
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
        self.warmup_steps = int(total_training_steps * warmup_frac)
        self.total_env_steps = 0

        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        assert abs(self.model.opt.timestep * PHYSICS_SUBSTEPS * DECIMATION - self.step_dt) < 1e-6
        self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)]
        m = self.model
        self.hand_id = m.body("hand").id
        self.tip_site = m.site("pipette_tip").id
        self.beaker_geom_id = m.geom("beaker_geom").id
        self.dish_geom_id = m.geom("dish_geom").id
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
                if (g1 == self.beaker_geom_id or g2 == self.beaker_geom_id or
                    g1 == self.dish_geom_id or g2 == self.dish_geom_id):
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
        take = np.where(in_beaker, FLOW_ML_PER_STEP, 0.0)
        take = np.minimum(take, np.minimum(self.beaker_ml, PIPETTE_CAPACITY_ML - self.pipette_ml))
        self.beaker_ml -= take; self.pipette_ml += take
        over_dish = (np.linalg.norm(tip[:, :2] - self.dish_pos[:, :2], axis=1) < DISH_RADIUS - 0.01) & (tip[:, 2] < DISPENSE_MAX_TIP_Z)
        give = np.where(over_dish, FLOW_ML_PER_STEP, 0.0)
        give = np.minimum(give, np.minimum(self.pipette_ml, DISH_CAPACITY_ML - self.dish_ml))
        before = np.minimum(self.dish_ml, self.target_ml)
        self.pipette_ml -= give; self.dish_ml += give
        self.progress = np.minimum(self.dish_ml, self.target_ml) - before

    def goal_distance(self):
        beaker_goal = self.beaker_pos.copy(); beaker_goal[:, 2] = 0.08
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
        self.terms_done["time_out"] = self.episode_length_buf.numpy() >= self.max_episode_length
        self.terms_done["success"] = self.dish_at_target()
        self.terms_done["overfill"] = self.dish_overfilled()
        self.terms_done["tip_below_table"] = self.tip_below_table()
        if self.reward_version == "v6":
            if self._in_warmup():
                self.terms_done["integrity_violation"] = np.zeros(self.num_envs, dtype=bool)
            else:
                self.terms_done["integrity_violation"] = self.penetration_event_this_step.copy()
            self.terminated = (self.terms_done["success"] | self.terms_done["overfill"]
                               | self.terms_done["tip_below_table"] | self.terms_done["integrity_violation"])
        else:
            self.terminated = self.terms_done["success"] | self.terms_done["overfill"] | self.terms_done["tip_below_table"]
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
            r["penetration_term"] = np.where(self.terms_done["integrity_violation"], REWARD_PENETRATION_TERMINATE, 0.0)
            r["safe_transit"] = self._safe_transit_bonus()
        total = np.zeros(self.num_envs)
        for k, v in r.items():
            self.reward_sums[k] += v
            total += v
        return total

    def _command(self): return np.stack([self.target_ml, self.pipette_ml, self.dish_ml], axis=1)

    def _compute_obs(self):
        f32 = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=self.device)
        positions = np.concatenate([self.beaker_pos[:, :2], self.dish_pos[:, :2]], axis=1)
        joint_vel = np.stack([d.qvel[:9] for d in self.datas])
        self.obs_buf = {
            "policy": f32(np.concatenate([self.tip_pos, self._command(), self.action, positions], axis=1)),
            "critic": f32(np.concatenate([self.tip_pos, self._command(), positions, joint_vel, self.action], axis=1)),
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

        noise = self.rng.uniform(-RESET_JOINT_NOISE, RESET_JOINT_NOISE, size=(len(ids), 9))
        qpos = np.clip(HOME_QPOS + noise, self.joint_range[:, 0], self.joint_range[:, 1])
        b_off, d_off = self._sample_offsets(len(ids))
        self.beaker_offset[ids] = b_off; self.dish_offset[ids] = d_off
        for j, i in enumerate(ids):
            d = self.datas[i]
            d.qpos[:9] = qpos[j]; d.qvel[:] = 0.0
            d.ctrl[:7] = qpos[j, :7]; d.ctrl[7] = 255.0
            d.mocap_pos[self.beaker_mocap] = np.array(BEAKER_POS) + np.array([b_off[j, 0], b_off[j, 1], 0.0])
            d.mocap_pos[self.dish_mocap] = np.array(DISH_POS) + np.array([d_off[j, 0], d_off[j, 1], 0.0])
            mujoco.mj_forward(self.model, d)
        self.target_ml[ids] = self.rng.uniform(*TARGET_RANGE_ML, size=len(ids))
        self.pipette_ml[ids] = 0.0; self.dish_ml[ids] = 0.0; self.beaker_ml[ids] = BEAKER_INITIAL_ML
        self.progress[ids] = 0.0; self.target_pos[ids] = HOME_TIP_POS
        self.action[ids] = 0.0; self.prev_action[ids] = 0.0
        self.episode_length_buf[torch.as_tensor(ids)] = 0
        self._refresh_poses(ids)
        self.contact_force_this_step[ids] = 0.0
        self.penetration_event_this_step[ids] = False
        self.penetration_count_this_ep[ids] = 0
        self.max_contact_force_this_ep[ids] = 0.0
        self.safe_transit_this_ep[ids] = 0.0
        self.prev_dist_beaker[ids] = np.linalg.norm(self.tip_pos[ids, :2] - self.beaker_pos[ids, :2], axis=1)
        self.prev_tip_z[ids] = self.tip_pos[ids, 2]
        self.prev_tip_xy[ids] = self.tip_pos[ids, :2]

    def reset(self):
        self._reset_idx(np.arange(self.num_envs))
        self.extras = {}
        return self._compute_obs(), self.extras

    def step(self, actions):
        a = actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions)
        self.prev_action[:] = self.action
        self.action[:] = np.clip(a, -1.0, 1.0)
        self.target_pos += self.action * STEP_SIZE
        np.clip(self.target_pos, self.ws_min, self.ws_max, out=self.target_pos)
        self._apply_actions()
        if self.viewer is not None: self.viewer.sync()
        self.episode_length_buf += 1
        self.total_env_steps += self.num_envs
        self._update_safety()
        self._compute_terminations()
        reward = self._compute_rewards()
        self.extras = {}
        if self.auto_reset:
            self._reset_idx(np.nonzero(self.terminated | self.time_outs)[0])
        self._update_fluid()
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
        "actor": mlp({"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}),
        "critic": mlp(None),
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 5, "num_mini_batches": 4,
                      "learning_rate": 3.0e-4, "schedule": "adaptive", "gamma": 0.99, "lam": 0.95,
                      "entropy_coef": 0.01, "desired_kl": 0.01, "max_grad_norm": 1.0,
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
    phase = np.zeros(n, dtype=int); done_counts = {}; ep_lens = []; ep_len = np.zeros(n)
    t = t0 = time.perf_counter()
    for step in range(max_steps):
        if not env.viewer_running(): break
        b_g, d_g = env.beaker_pos.copy(), env.dish_pos.copy()
        hb, hd = b_g.copy(), d_g.copy(); hb[:, 2] = 0.25; b_g[:, 2] = 0.08; hd[:, 2] = 0.20; d_g[:, 2] = 0.06
        goal = np.stack([hb, b_g, hb, hd, d_g, hd], axis=1)[np.arange(n), phase]
        _, rew, terminated, truncated, _ = env.step(np.clip((goal - env.target_pos) / STEP_SIZE, -1.0, 1.0))
        ep_len += 1
        reached = np.linalg.norm(env.tip_pos - goal, axis=1) < 0.015
        need = np.minimum(env.target_ml - env.dish_ml, PIPETTE_CAPACITY_ML)
        reached &= (phase != 1) | (env.pipette_ml >= need - 1e-3)
        reached &= (phase != 4) | (env.dish_ml >= env.target_ml - SUCCESS_TOL_ML)
        phase = np.where(reached, (phase + 1) % 6, phase)
        dones = (terminated | truncated).cpu().numpy()
        if dones.any():
            phase[dones] = 0; ep_lens += ep_len[dones].tolist(); ep_len[dones] = 0
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
                if env.dish_at_target()[0]: outcome = "success"
                elif env.dish_overfilled()[0]: outcome = "overfill"
                elif env.tip_below_table()[0]: outcome = "tip_below_table"
                elif REWARD_VERSION == "v6" and env.penetration_event_this_step[0]: outcome = "penetration_event"
                elif step + 1 >= hold_limit: outcome = "time_out"
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
        if step % 50 == 0:
            print(f"[step {step:4d}] dish0={env.dish_ml[0]:.2f} pen0={env.penetration_count_this_ep[0]}")
        if hold:
            if env.dish_at_target()[0]: outcome = "success"
            elif env.dish_overfilled()[0]: outcome = "overfill"
            elif env.tip_below_table()[0]: outcome = "tip_below_table"
            elif REWARD_VERSION == "v6" and env.penetration_event_this_step[0]: outcome = "penetration_event"
            elif step + 1 >= hold_limit: outcome = "time_out"
            if outcome:
                print(f"[Hold] ended at step {step+1}: {outcome}; target={env.target_ml[0]:.2f} "
                      f"dish={env.dish_ml[0]:.2f} pen={env.penetration_count_this_ep[0]} "
                      f"max_cf={env.max_contact_force_this_ep[0]:.1f}N")
                break
        t = _pace(t)
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
        device=device, threads=args_cli.threads, auto_reset=not args_cli.hold,
        total_training_steps=total_training_steps,
        reward_version=REWARD_VERSION,
        warmup_frac=WARMUP_FRAC,
    )
    if args_cli.viz: env.launch_viewer()
    print(f"*** model 3 v6 (MuJoCo {mujoco.__version__}): {args_cli.num_envs} envs, mode={args_cli.mode}, "
          f"reward={REWARD_VERSION}, warmup_steps={env.warmup_steps}, torch device {device} ***")
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
