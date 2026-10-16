"""
Model 3 (MuJoCo): PPO with known container positions, both the beaker and the dish random every episode.

Forked from ../model_2_mujoco_teacher_student_random_dish_positions/liquid_transfer_mj.py. Differences:

  * No camera, no ResNet, no teacher-student split. The policy is told the true beaker and dish positions
    (what model 2's teacher saw) and is trained directly with PPO.
  * The beaker moves too: beaker at (0.40, -0.20) + U(-beaker_range, beaker_range)^2 and dish at
    (0.40, 0.20) + U(-dish_range, dish_range)^2 each episode (default range 0.20 m). Positions closer than 0.29 m
    to the robot base are redrawn (unreachable with a vertical pipette), and so are pairs whose centres are
    closer than 0.18 m (the containers would overlap: radii 0.05 + 0.08).
  * Workspace box widened to y >= -0.45 so a beaker at y = -0.40 is reachable.

Everything else is model 2: the liquid model, reward v4, terminations, the held-target IK action, 30 Hz control.

Observation groups:
  policy  tip position (3), order / pipette / dish ml (3), last action (3), beaker xy (2), dish xy (2) = 13
  critic  tip (3), command (3), beaker xy (2), dish xy (2), joint velocities (9), last action (3) = 22

Run from this folder with the Isaac Lab venv (it has mujoco, torch, rsl_rl, tensorboard):
    PY=../IsaacLab/.venv/bin/python
    Checks:    $PY liquid_transfer_m3.py --mode probe --num_envs 4
               $PY liquid_transfer_m3.py --mode scripted --num_envs 16 --max_steps 900
    Train:     $PY liquid_transfer_m3.py --mode train --num_envs 128 --max_iterations 2000 --run_name m3
    Evaluate:  $PY liquid_transfer_m3.py --mode play --num_envs 32 --max_steps 1800 --seed 7 --checkpoint policy.pt
    Watch:     add --num_envs 1 --hold --viz
"""

from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

parser = argparse.ArgumentParser(description="Liquid transfer, model 3 in MuJoCo: PPO, known positions, random beaker and dish.")
parser.add_argument("--mode", choices=["random", "probe", "scripted", "train", "play"], default="random")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--max_steps", type=int, default=300, help="Steps to run in random/scripted/play mode.")
parser.add_argument("--max_iterations", type=int, default=300, help="Training iterations.")
parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint .pt to play, or to resume training from.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--episode_length_s", type=float, default=15.0, help="Time limit per episode in seconds.")
parser.add_argument("--dish_range", type=float, default=0.20, help="Dish placed uniformly within +-this (m) of nominal.")
parser.add_argument("--beaker_range", type=float, default=0.20, help="Beaker placed uniformly within +-this (m) of nominal.")
parser.add_argument("--dish_offset", type=str, default="", help="Fixed dish offset 'dx,dy' (m) from nominal (0.4, 0.2). Use '='.")
parser.add_argument("--beaker_offset", type=str, default="", help="Fixed beaker offset 'dx,dy' (m) from nominal (0.4, -0.2). Use '='.")
parser.add_argument("--hold", action="store_true", help="Play mode: one episode without auto-reset, then keep the window open.")
parser.add_argument("--viz", action="store_true", help="Open the MuJoCo viewer on env 0 (random/scripted/play modes).")
parser.add_argument("--device", type=str, default="cuda", help="Torch device for the RL networks (the physics is always CPU).")
parser.add_argument("--threads", type=int, default=0, help="Physics threads (0 = min(8, cpu count)).")
args_cli = parser.parse_args()

if not args_cli.viz:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(HERE, "liquid_transfer_scene.xml")

# ----------------------------------------------------------------------------------------------
# Geometry (metres, robot base frame = MuJoCo world frame)
# ----------------------------------------------------------------------------------------------
BEAKER_POS = (0.4, -0.2, 0.06)  # nominal; radius 0.05, height 0.12 -> rim at z = 0.12
BEAKER_RADIUS = 0.05
DISH_POS = (0.4, 0.2, 0.01)  # nominal; radius 0.08, height 0.02
DISH_RADIUS = 0.08
MIN_REACH_M = 0.29  # container centres closer than this to the base are redrawn
MIN_SEPARATION_M = 0.18  # beaker-dish centre distance below this is redrawn (0.05 + 0.08 radii + margin)
DISTANCE_BINS_CM = (5, 10, 15, 20)
TIP_DOWN_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])

# Liquid model (millilitres)
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
WORKSPACE_MIN = (0.15, -0.45, 0.03)  # model 2: y >= -0.35; the beaker can now sit at y = -0.40
WORKSPACE_MAX = (0.65, 0.45, 0.45)
IK_LAMBDA = 0.01
HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
RESET_JOINT_NOISE = 0.1

# Reward v4, per-step values
REWARD_PROGRESS_PER_ML = 10.0
REWARD_SUCCESS = 100.0
REWARD_REACH_PER_M = -1.0
REWARD_TIME = -0.05
REWARD_OVERFILL = -20.0
REWARD_ACTION_RATE = -0.01
REWARD_TERMS = ("progress", "success", "reach", "time", "overfill", "action_rate")
TERMINATION_TERMS = ("time_out", "success", "overfill", "tip_below_table")


# ----------------------------------------------------------------------------------------------
# Quaternion helpers (w, x, y, z), batched
# ----------------------------------------------------------------------------------------------
def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def axis_angle_from_quat(q: np.ndarray) -> np.ndarray:
    q = np.where(q[:, :1] < 0, -q, q)
    v = q[:, 1:]
    s = np.linalg.norm(v, axis=1)
    angle = 2.0 * np.arctan2(s, q[:, 0])
    scale = np.where(s > 1e-8, angle / np.maximum(s, 1e-12), 2.0)
    return v * scale[:, None]


def distance_bin(dist_m: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.array(DISTANCE_BINS_CM) / 100.0, dist_m, side="right")


def distance_bin_label(b: int) -> str:
    lo = 0 if b == 0 else DISTANCE_BINS_CM[b - 1]
    return f"{lo}+cm" if b >= len(DISTANCE_BINS_CM) else f"{lo}-{DISTANCE_BINS_CM[b]}cm"


# ----------------------------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------------------------
class LiquidTransferM3Env:
    """N independent MuJoCo worlds sharing one model, stepped in lock-step. Same step order as Isaac Lab's
    ManagerBasedRLEnv (terminations and rewards read the fluid state of the previous update)."""

    def __init__(
        self,
        num_envs: int,
        seed: int = 42,
        episode_length_s: float = 15.0,
        dish_range: float = 0.20,
        beaker_range: float = 0.20,
        fixed_dish_offset: tuple[float, float] | None = None,
        fixed_beaker_offset: tuple[float, float] | None = None,
        device: str = "cpu",
        threads: int = 0,
        auto_reset: bool = True,
    ):
        self.num_envs = num_envs
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.dish_range, self.beaker_range = dish_range, beaker_range
        self.fixed_dish_offset, self.fixed_beaker_offset = fixed_dish_offset, fixed_beaker_offset
        self.auto_reset = auto_reset
        self.step_dt = 1.0 / CONTROL_HZ
        self.episode_length_s = episode_length_s
        self.max_episode_length = math.ceil(episode_length_s / self.step_dt)

        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        assert abs(self.model.opt.timestep * PHYSICS_SUBSTEPS * DECIMATION - self.step_dt) < 1e-6, "scene timestep changed"
        self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)]
        m = self.model
        self.hand_id = m.body("hand").id
        self.tip_site = m.site("pipette_tip").id
        self.beaker_mocap = m.body_mocapid[m.body("beaker").id]
        self.dish_mocap = m.body_mocapid[m.body("petri_dish").id]
        self.joint_range = m.jnt_range.copy()
        assert np.allclose(m.key_qpos[m.key("home").id], HOME_QPOS)

        n = num_envs
        self.target_pos = np.tile(np.array(HOME_TIP_POS), (n, 1))
        self.ws_min, self.ws_max = np.array(WORKSPACE_MIN), np.array(WORKSPACE_MAX)
        self.action = np.zeros((n, 3))
        self.prev_action = np.zeros((n, 3))
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))
        self.target_ml = np.zeros(n)
        self.pipette_ml = np.zeros(n)
        self.dish_ml = np.zeros(n)
        self.beaker_ml = np.full(n, BEAKER_INITIAL_ML)
        self.progress = np.zeros(n)
        self.tip_pos = np.zeros((n, 3))
        self.beaker_pos = np.zeros((n, 3))
        self.dish_pos = np.zeros((n, 3))
        self.dish_offset = np.zeros((n, 2))
        self.beaker_offset = np.zeros((n, 2))
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.terms_done = {k: np.zeros(n, dtype=bool) for k in TERMINATION_TERMS}
        self.terminated = np.zeros(n, dtype=bool)
        self.time_outs = np.zeros(n, dtype=bool)
        self.reward_sums = {k: np.zeros(n) for k in REWARD_TERMS}
        self.extras: dict = {}
        self.obs_buf: dict[str, torch.Tensor] = {}
        self.pool = ThreadPoolExecutor(max_workers=threads or min(8, os.cpu_count() or 1))
        self.viewer = None
        self.reset()

    # ---------------------------------------------------------------- IK + physics
    def _ik_tick(self):
        n = self.num_envs
        jac = np.zeros((n, 6, 7))
        err = np.zeros((n, 6))
        cur_quat = np.zeros((n, 4))
        for i, d in enumerate(self.datas):
            mujoco.mj_jacSite(self.model, d, self._jacp, self._jacr, self.tip_site)
            jac[i, :3] = self._jacp[:, :7]
            jac[i, 3:] = self._jacr[:, :7]
            err[i, :3] = self.target_pos[i] - d.site_xpos[self.tip_site]
            cur_quat[i] = d.xquat[self.hand_id]
        q_err = quat_mul(np.tile(TIP_DOWN_QUAT_WXYZ, (n, 1)), quat_conj(cur_quat))
        err[:, 3:] = axis_angle_from_quat(q_err)
        jjt = jac @ jac.transpose(0, 2, 1) + (IK_LAMBDA**2) * np.eye(6)
        dq = jac.transpose(0, 2, 1) @ np.linalg.solve(jjt, err[:, :, None])
        for i, d in enumerate(self.datas):
            d.ctrl[:7] = d.qpos[:7] + dq[i, :, 0]
            d.ctrl[7] = 255.0

    def _step_physics(self, idx: int):
        d = self.datas[idx]
        for _ in range(PHYSICS_SUBSTEPS):
            mujoco.mj_step(self.model, d)

    def _apply_actions(self):
        for _ in range(DECIMATION):
            self._ik_tick()
            list(self.pool.map(self._step_physics, range(self.num_envs)))

    # ---------------------------------------------------------------- fluid
    def _refresh_poses(self, ids=None):
        for i in range(self.num_envs) if ids is None else ids:
            d = self.datas[i]
            self.tip_pos[i] = d.site_xpos[self.tip_site]
            self.beaker_pos[i] = d.mocap_pos[self.beaker_mocap]
            self.dish_pos[i] = d.mocap_pos[self.dish_mocap]

    def _update_fluid(self):
        self._refresh_poses()
        tip = self.tip_pos
        in_beaker = (np.linalg.norm(tip[:, :2] - self.beaker_pos[:, :2], axis=1) < BEAKER_RADIUS - 0.005) & (
            tip[:, 2] < ASPIRATE_MAX_TIP_Z
        )
        take = np.where(in_beaker, FLOW_ML_PER_STEP, 0.0)
        take = np.minimum(take, np.minimum(self.beaker_ml, PIPETTE_CAPACITY_ML - self.pipette_ml))
        self.beaker_ml -= take
        self.pipette_ml += take
        over_dish = (np.linalg.norm(tip[:, :2] - self.dish_pos[:, :2], axis=1) < DISH_RADIUS - 0.01) & (
            tip[:, 2] < DISPENSE_MAX_TIP_Z
        )
        give = np.where(over_dish, FLOW_ML_PER_STEP, 0.0)
        give = np.minimum(give, np.minimum(self.pipette_ml, DISH_CAPACITY_ML - self.dish_ml))
        before = np.minimum(self.dish_ml, self.target_ml)
        self.pipette_ml -= give
        self.dish_ml += give
        self.progress = np.minimum(self.dish_ml, self.target_ml) - before

    # ---------------------------------------------------------------- MDP terms
    def goal_distance(self) -> np.ndarray:
        beaker_goal = self.beaker_pos.copy()
        beaker_goal[:, 2] = 0.08
        dish_goal = self.dish_pos.copy()
        dish_goal[:, 2] = 0.06
        to_dish = np.linalg.norm(self.tip_pos - dish_goal, axis=1)
        via_beaker = np.linalg.norm(self.tip_pos - beaker_goal, axis=1) + np.linalg.norm(beaker_goal - dish_goal, axis=1)
        return np.where(self.pipette_ml > 0.05, to_dish, via_beaker)

    def dish_at_target(self) -> np.ndarray:
        return np.abs(self.dish_ml - self.target_ml) < SUCCESS_TOL_ML

    def dish_overfilled(self) -> np.ndarray:
        return self.dish_ml > self.target_ml + SUCCESS_TOL_ML

    def tip_below_table(self) -> np.ndarray:
        return self.tip_pos[:, 2] < 0.0

    def _compute_terminations(self):
        self.terms_done["time_out"] = self.episode_length_buf.numpy() >= self.max_episode_length
        self.terms_done["success"] = self.dish_at_target()
        self.terms_done["overfill"] = self.dish_overfilled()
        self.terms_done["tip_below_table"] = self.tip_below_table()
        self.time_outs = self.terms_done["time_out"].copy()
        self.terminated = self.terms_done["success"] | self.terms_done["overfill"] | self.terms_done["tip_below_table"]

    def _compute_rewards(self) -> np.ndarray:
        r = {
            "progress": REWARD_PROGRESS_PER_ML * self.progress,
            "success": REWARD_SUCCESS * self.dish_at_target(),
            "reach": REWARD_REACH_PER_M * self.goal_distance(),
            "time": REWARD_TIME * (~self.terminated),
            "overfill": REWARD_OVERFILL * self.dish_overfilled(),
            "action_rate": REWARD_ACTION_RATE * np.sum((self.action - self.prev_action) ** 2, axis=1),
        }
        total = np.zeros(self.num_envs)
        for k, v in r.items():
            self.reward_sums[k] += v
            total += v
        return total

    # ---------------------------------------------------------------- observations
    def _command(self) -> np.ndarray:
        return np.stack([self.target_ml, self.pipette_ml, self.dish_ml], axis=1)

    def _compute_obs(self) -> dict[str, torch.Tensor]:
        f32 = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=self.device)
        positions = np.concatenate([self.beaker_pos[:, :2], self.dish_pos[:, :2]], axis=1)
        joint_vel = np.stack([d.qvel[:9] for d in self.datas])
        self.obs_buf = {
            "policy": f32(np.concatenate([self.tip_pos, self._command(), self.action, positions], axis=1)),
            "critic": f32(np.concatenate([self.tip_pos, self._command(), positions, joint_vel, self.action], axis=1)),
        }
        return self.obs_buf

    # ---------------------------------------------------------------- reset
    def _sample_offsets(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Beaker and dish offsets from nominal for k envs: fixed where given, else uniform, redrawn while either is
        too close to the base or the two containers would overlap."""
        b_nom, d_nom = np.array(BEAKER_POS[:2]), np.array(DISH_POS[:2])
        draw_b = lambda m: np.tile(np.array(self.fixed_beaker_offset), (m, 1)) if self.fixed_beaker_offset is not None else self.rng.uniform(-self.beaker_range, self.beaker_range, size=(m, 2))
        draw_d = lambda m: np.tile(np.array(self.fixed_dish_offset), (m, 1)) if self.fixed_dish_offset is not None else self.rng.uniform(-self.dish_range, self.dish_range, size=(m, 2))
        b, d = draw_b(k), draw_d(k)
        for _ in range(100):
            bad = (np.linalg.norm(b_nom + b, axis=1) < MIN_REACH_M) | (np.linalg.norm(d_nom + d, axis=1) < MIN_REACH_M)
            bad |= np.linalg.norm((b_nom + b) - (d_nom + d), axis=1) < MIN_SEPARATION_M
            if not bad.any():
                break
            nb = int(bad.sum())
            b[bad], d[bad] = draw_b(nb), draw_d(nb)
        else:
            raise RuntimeError("could not sample valid container positions (fixed offsets unreachable or overlapping?)")
        return b, d

    def _reset_idx(self, ids: np.ndarray):
        if len(ids) == 0:
            return
        log = {}
        for k in REWARD_TERMS:
            log[f"Episode_Reward/{k}"] = float(self.reward_sums[k][ids].mean() / self.episode_length_s)
            self.reward_sums[k][ids] = 0.0
        for k in TERMINATION_TERMS:
            log[f"Episode_Termination/{k}"] = float(np.count_nonzero(self.terms_done[k][ids]))
        log["Metrics/fluid/dish_error_ml"] = float(np.abs(self.dish_ml[ids] - self.target_ml[ids]).mean())
        log["Metrics/fluid/pipette_ml"] = float(self.pipette_ml[ids].mean())
        success = self.terms_done["success"][ids].astype(float)
        for name, off in (("dish", self.dish_offset), ("beaker", self.beaker_offset)):
            bins = distance_bin(np.linalg.norm(off[ids], axis=1))
            for b in np.unique(bins).tolist():
                log[f"Metrics/fluid/success_{name}_{distance_bin_label(b)}"] = float(success[bins == b].mean())
        self.extras["log"] = log

        noise = self.rng.uniform(-RESET_JOINT_NOISE, RESET_JOINT_NOISE, size=(len(ids), 9))
        qpos = np.clip(HOME_QPOS + noise, self.joint_range[:, 0], self.joint_range[:, 1])
        b_off, d_off = self._sample_offsets(len(ids))
        self.beaker_offset[ids] = b_off
        self.dish_offset[ids] = d_off
        for j, i in enumerate(ids):
            d = self.datas[i]
            d.qpos[:9] = qpos[j]
            d.qvel[:] = 0.0
            d.ctrl[:7] = qpos[j, :7]
            d.ctrl[7] = 255.0
            d.mocap_pos[self.beaker_mocap] = np.array(BEAKER_POS) + np.array([b_off[j, 0], b_off[j, 1], 0.0])
            d.mocap_pos[self.dish_mocap] = np.array(DISH_POS) + np.array([d_off[j, 0], d_off[j, 1], 0.0])
            mujoco.mj_forward(self.model, d)
        self.target_ml[ids] = self.rng.uniform(*TARGET_RANGE_ML, size=len(ids))
        self.pipette_ml[ids] = 0.0
        self.dish_ml[ids] = 0.0
        self.beaker_ml[ids] = BEAKER_INITIAL_ML
        self.progress[ids] = 0.0
        self.target_pos[ids] = HOME_TIP_POS
        self.action[ids] = 0.0
        self.prev_action[ids] = 0.0
        self.episode_length_buf[torch.as_tensor(ids)] = 0
        self._refresh_poses(ids)

    def reset(self):
        self._reset_idx(np.arange(self.num_envs))
        self.extras = {}
        return self._compute_obs(), self.extras

    # ---------------------------------------------------------------- step
    def step(self, actions):
        a = actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions)
        self.prev_action[:] = self.action
        self.action[:] = np.clip(a, -1.0, 1.0)
        self.target_pos += self.action * STEP_SIZE
        np.clip(self.target_pos, self.ws_min, self.ws_max, out=self.target_pos)
        self._apply_actions()
        if self.viewer is not None:
            self.viewer.sync()
        self.episode_length_buf += 1
        self._compute_terminations()
        reward = self._compute_rewards()
        self.extras = {}
        if self.auto_reset:
            self._reset_idx(np.nonzero(self.terminated | self.time_outs)[0])
        self._update_fluid()
        obs = self._compute_obs()
        t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=self.device)
        return obs, t(reward, torch.float32), t(self.terminated, torch.bool), t(self.time_outs, torch.bool), self.extras

    # ---------------------------------------------------------------- diagnostics
    def pipette_tilt_deg(self) -> np.ndarray:
        z = np.stack([d.xmat[self.hand_id].reshape(3, 3)[:, 2] for d in self.datas])
        return np.degrees(np.arccos(np.clip(-z[:, 2], -1.0, 1.0)))

    def launch_viewer(self):
        import mujoco.viewer

        self.viewer = mujoco.viewer.launch_passive(self.model, self.datas[0])
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.viewer.cam.fixedcamid = self.model.cam("viewer").id
        return self.viewer

    def viewer_running(self) -> bool:
        return self.viewer is None or self.viewer.is_running()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.pool.shutdown()


# ----------------------------------------------------------------------------------------------
# RSL-RL glue
# ----------------------------------------------------------------------------------------------
class RslRlVecEnv:
    def __init__(self, env: LiquidTransferM3Env, clip_actions: float = 1.0):
        self.env = env
        self.unwrapped = env
        self.num_envs = env.num_envs
        self.num_actions = 3
        self.max_episode_length = env.max_episode_length
        self.device = env.device
        self.clip_actions = clip_actions
        self.cfg = {"task": "liquid_transfer_m3_mujoco", "num_envs": env.num_envs, "episode_length_s": env.episode_length_s}

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.env.episode_length_buf = value.cpu()

    def get_observations(self):
        from tensordict import TensorDict

        return TensorDict(self.env.obs_buf, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor):
        from tensordict import TensorDict

        actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs, rew, terminated, truncated, extras = self.env.step(actions)
        extras["time_outs"] = truncated
        return TensorDict(obs, batch_size=[self.num_envs]), rew, (terminated | truncated).to(torch.long), extras

    def close(self):
        self.env.close()


def make_ppo_cfg(seed: int, device: str, max_iterations: int, run_name: str) -> dict:
    """PPO, same hyperparameters as model 2's teacher."""
    mlp = lambda dist: {"class_name": "MLPModel", "hidden_dims": [256, 128], "activation": "elu", "obs_normalization": True, "distribution_cfg": dist}
    return {
        "seed": seed,
        "device": device,
        "num_steps_per_env": 32,
        "init_at_random_ep_len": True,
        "max_iterations": max_iterations,
        "empirical_normalization": {},
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "clip_actions": 1.0,
        "check_for_nan": True,
        "save_interval": 25,
        "experiment_name": "liquid_transfer_m3",
        "run_name": run_name,
        "logger": "tensorboard",
        "resume": False,
        "load_run": ".*",
        "load_checkpoint": "model_.*.pt",
        "class_name": "OnPolicyRunner",
        "actor": mlp({"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}),
        "critic": mlp(None),
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 3.0e-4,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "entropy_coef": 0.01,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "optimizer": "adam",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "normalize_advantage_per_mini_batch": False,
            "share_cnn_encoders": False,
            "use_mixed_precision": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "torch_compile_mode": None,
        "policy": {},
    }


# ----------------------------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------------------------
def _pace(t_last: float) -> float:
    if args_cli.viz:
        time.sleep(max(0.0, t_last + 1.0 / CONTROL_HZ - time.perf_counter()))
    return time.perf_counter()


def _count_terminations(env, counts: dict):
    for name in TERMINATION_TERMS:
        k = int(env.terms_done[name].sum())
        if k:
            counts[name] = counts.get(name, 0) + k


def run_random(env: LiquidTransferM3Env, max_steps: int):
    obs, _ = env.reset()
    print("[Obs] " + ", ".join(f"{k}: {tuple(v.shape)}" for k, v in obs.items()))
    totals = np.zeros(env.num_envs)
    done_counts: dict = {}
    t = t0 = time.perf_counter()
    for step in range(max_steps):
        if not env.viewer_running():
            break
        _, rew, _, _, _ = env.step(env.rng.uniform(-1.0, 1.0, size=(env.num_envs, 3)))
        totals += rew.cpu().numpy()
        _count_terminations(env, done_counts)
        if step % 100 == 0:
            print(f"[step {step:4d}] tip0={env.tip_pos[0].round(3).tolist()} beaker0={env.beaker_pos[0, :2].round(2).tolist()} "
                  f"dish0={env.dish_pos[0, :2].round(2).tolist()} reward0={rew[0]:.3f} tilt_deg(max)={env.pipette_tilt_deg().max():.1f}")
        t = _pace(t)
    wall = time.perf_counter() - t0
    print(f"[Random] mean total reward over {max_steps} steps: {totals.mean():.2f}; terminations: {done_counts}; "
          f"{wall:.1f} s wall = {max_steps * env.num_envs / wall:.0f} env-steps/s")


def run_probe(env: LiquidTransferM3Env):
    """Drive the held target to each env's beaker fill point, then dish point; report the reach error."""
    env.reset()
    env.auto_reset = False  # a success mid-drive would otherwise reset the arm and spoil the reach measurement
    env.step(np.zeros((env.num_envs, 3)))
    for name, goal_z, get_pos in (("beaker fill point", 0.08, lambda: env.beaker_pos), ("dish dispense point", 0.06, lambda: env.dish_pos)):
        for _ in range(150):
            goal = get_pos().copy()
            goal[:, 2] = goal_z
            env.step(np.clip((goal - env.target_pos) / STEP_SIZE, -1.0, 1.0))
        goal = get_pos().copy()
        goal[:, 2] = goal_z
        err = np.linalg.norm(env.tip_pos - goal, axis=1)
        tilt = env.pipette_tilt_deg()
        for i in range(env.num_envs):
            print(f"[Probe] env {i} beaker offset {env.beaker_offset[i].round(2).tolist()} dish offset {env.dish_offset[i].round(2).tolist()} "
                  f"-> {name} {goal[i].round(2).tolist()}: tip {env.tip_pos[i].round(3).tolist()} error {100 * err[i]:.1f} cm "
                  f"tilt {tilt[i]:.1f} deg pipette {env.pipette_ml[i]:.1f} ml dish {env.dish_ml[i]:.1f} ml")


def run_scripted(env: LiquidTransferM3Env, max_steps: int):
    env.reset()
    n = env.num_envs
    phase = np.zeros(n, dtype=int)
    totals = np.zeros(n)
    done_counts: dict = {}
    episode_lengths: list = []
    ep_len = np.zeros(n)
    t = t0 = time.perf_counter()
    for step in range(max_steps):
        if not env.viewer_running():
            break
        beaker_goal, dish_goal = env.beaker_pos.copy(), env.dish_pos.copy()
        high_b, high_d = beaker_goal.copy(), dish_goal.copy()
        high_b[:, 2], beaker_goal[:, 2] = 0.25, 0.08
        high_d[:, 2], dish_goal[:, 2] = 0.20, 0.06
        goal = np.stack([high_b, beaker_goal, high_b, high_d, dish_goal, high_d], axis=1)[np.arange(n), phase]
        _, rew, terminated, truncated, _ = env.step(np.clip((goal - env.target_pos) / STEP_SIZE, -1.0, 1.0))
        totals += rew.cpu().numpy()
        ep_len += 1
        reached = np.linalg.norm(env.tip_pos - goal, axis=1) < 0.015
        need = np.minimum(env.target_ml - env.dish_ml, PIPETTE_CAPACITY_ML)
        reached &= (phase != 1) | (env.pipette_ml >= need - 1e-3)
        reached &= (phase != 4) | (env.dish_ml >= env.target_ml - SUCCESS_TOL_ML)
        phase = np.where(reached, (phase + 1) % 6, phase)
        dones = (terminated | truncated).cpu().numpy()
        if dones.any():
            phase[dones] = 0
            episode_lengths += ep_len[dones].tolist()
            ep_len[dones] = 0
        _count_terminations(env, done_counts)
        if step % 50 == 0:
            print(f"[step {step:4d}] env0 beaker={env.beaker_pos[0, :2].round(2).tolist()} dish={env.dish_pos[0, :2].round(2).tolist()} "
                  f"phase={int(phase[0])} target={env.target_ml[0]:.2f} pipette={env.pipette_ml[0]:.2f} dish_ml={env.dish_ml[0]:.2f} reward={rew[0]:.3f}")
        t = _pace(t)
    wall = time.perf_counter() - t0
    mean_len = sum(episode_lengths) / max(len(episode_lengths), 1)
    print(f"[Scripted] terminations over {max_steps} steps x {n} envs: {done_counts}; mean episode length {mean_len:.0f} steps; "
          f"mean reward per step {totals.mean() / max_steps:.3f}; {wall:.1f} s wall")


def run_train(env: LiquidTransferM3Env, agent_cfg: dict, log_dir: str, resume_path: str):
    from rsl_rl.runners import OnPolicyRunner

    runner = OnPolicyRunner(RslRlVecEnv(env, agent_cfg["clip_actions"]), agent_cfg, log_dir=log_dir, device=agent_cfg["device"])
    if resume_path:
        runner.load(resume_path, map_location=agent_cfg["device"])
    print(f"[Train] logging to {log_dir}")
    runner.learn(num_learning_iterations=agent_cfg["max_iterations"], init_at_random_ep_len=True)


def run_play(env: LiquidTransferM3Env, agent_cfg: dict, checkpoint: str, max_steps: int, hold: bool = False):
    from rsl_rl.runners import OnPolicyRunner

    venv = RslRlVecEnv(env, agent_cfg["clip_actions"])
    runner = OnPolicyRunner(venv, agent_cfg, log_dir=None, device=agent_cfg["device"])
    runner.load(checkpoint, map_location=agent_cfg["device"])
    policy = runner.get_inference_policy(device=env.device)
    obs = venv.get_observations()
    n = env.num_envs
    counts: dict = {}
    per_bin: dict = {"dish": {}, "beaker": {}}
    ep_lens: list = []
    ep_len = np.zeros(n)
    hold_limit = int(round(args_cli.episode_length_s * CONTROL_HZ))
    outcome = None
    t = t0 = time.perf_counter()
    with torch.inference_mode():
        for step in range(max_steps):
            if not env.viewer_running():
                break
            dish_before, beaker_before = env.dish_offset.copy(), env.beaker_offset.copy()
            obs, _, dones, _ = venv.step(policy(obs))
            ep_len += 1
            _count_terminations(env, counts)
            done_ids = np.nonzero(dones.cpu().numpy())[0]
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
                print(f"[step {step:4d}] target0={env.target_ml[0]:.2f} pipette0={env.pipette_ml[0]:.2f} dish0={env.dish_ml[0]:.2f}")
            if hold:
                if env.dish_at_target()[0]:
                    outcome = "success"
                elif env.dish_overfilled()[0]:
                    outcome = "overfill"
                elif env.tip_below_table()[0]:
                    outcome = "tip_below_table"
                elif step + 1 >= hold_limit:
                    outcome = "time_out"
                if outcome:
                    print(f"[Hold] episode ended at step {step + 1}: {outcome}; beaker offset {env.beaker_offset[0].round(3).tolist()} m, "
                          f"dish offset {env.dish_offset[0].round(3).tolist()} m; target={env.target_ml[0]:.2f} "
                          f"pipette={env.pipette_ml[0]:.2f} dish={env.dish_ml[0]:.2f} ml")
                    break
            t = _pace(t)
    wall = time.perf_counter() - t0
    if hold:
        if outcome is None:
            print(f"[Hold] --max_steps {max_steps} reached before the episode ended")
        if env.viewer is None:
            print("[Hold] no window open (headless), nothing to hold")
            return
        print("[Hold] holding the final frame; close the window to exit")
        while env.viewer.is_running():
            env.viewer.sync()
            time.sleep(0.05)
        return
    episodes = sum(counts.values())
    rate = counts.get("success", 0) / max(episodes, 1)
    mean_len = sum(ep_lens) / max(len(ep_lens), 1)
    print(f"[Play] {max_steps} steps x {n} envs: {episodes} episodes, outcomes {counts}, success rate {rate:.1%}, "
          f"mean episode length {mean_len:.0f} steps; {wall:.1f} s wall")
    for name in ("dish", "beaker"):
        for b in sorted(per_bin[name]):
            s, k = per_bin[name][b]
            print(f"[Play] {name} {distance_bin_label(b)} from nominal: {s}/{k} = {s / max(k, 1):.1%}")


def _parse_offset(s: str, what: str):
    if not s:
        return None
    off = tuple(float(v) for v in s.split(","))
    print(f"[{what}] fixed offset {off} m from nominal")
    return off


def main():
    mode = args_cli.mode
    device = args_cli.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[Device] CUDA not available, using cpu")
        device = "cpu"
    env = LiquidTransferM3Env(
        num_envs=args_cli.num_envs,
        seed=args_cli.seed,
        episode_length_s=1.0e6 if args_cli.hold else args_cli.episode_length_s,
        dish_range=args_cli.dish_range,
        beaker_range=args_cli.beaker_range,
        fixed_dish_offset=_parse_offset(args_cli.dish_offset, "Dish"),
        fixed_beaker_offset=_parse_offset(args_cli.beaker_offset, "Beaker"),
        device=device,
        threads=args_cli.threads,
        auto_reset=not args_cli.hold,
    )
    if args_cli.viz:
        env.launch_viewer()
    print(f"*** Liquid transfer (model 3, MuJoCo {mujoco.__version__}) env ready: {env.num_envs} envs, mode={mode}, torch device {device} ***")
    try:
        if mode == "random":
            run_random(env, args_cli.max_steps)
        elif mode == "probe":
            run_probe(env)
        elif mode == "scripted":
            run_scripted(env, args_cli.max_steps)
        else:
            torch.manual_seed(args_cli.seed)
            agent_cfg = make_ppo_cfg(args_cli.seed, device, args_cli.max_iterations, args_cli.run_name)
            if mode == "train":
                run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{args_cli.run_name}" if args_cli.run_name else "")
                run_train(env, agent_cfg, os.path.join(HERE, "logs", "rsl_rl", agent_cfg["experiment_name"], run_name), args_cli.checkpoint)
            else:
                if not args_cli.checkpoint:
                    raise SystemExit("play needs --checkpoint <.pt>")
                run_play(env, agent_cfg, args_cli.checkpoint, args_cli.max_steps, hold=args_cli.hold)
    finally:
        env.close()


if __name__ == "__main__":
    main()
