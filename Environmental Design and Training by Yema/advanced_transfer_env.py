import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer


class AdvancedLiquidTransferEnv(gym.Env):
    """Advanced Multimodal Environment for Robotic Liquid Handling."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode="human"):
        super(AdvancedLiquidTransferEnv, self).__init__()
        self.render_mode = render_mode

        self.xml_model = """
        <mujoco model="advanced_lab_automation">
            <option integrator="RK4" timestep="0.002" gravity="0 0 -9.81"/>

            <worldbody>
                <geom name="table" type="box" size="0.5 0.5 0.4" pos="0.5 0.5 0.4" rgba="0.4 0.4 0.4 1"/>
                <light directional="true" diffuse=".8 .8 .8" pos="0.5 0.5 3" dir="0 0 -1"/>

                <body name="beaker" pos="0.2 0.3 0.82">
                    <geom name="beaker_geom" type="cylinder" size="0.06 0.06" rgba="0.7 0.7 0.9 0.5"/>
                </body>
                <body name="petri_dish" pos="0.8 0.7 0.81">
                    <geom name="petri_geom" type="cylinder" size="0.08 0.01" rgba="0.8 0.9 1.0 0.6"/>
                </body>

                <body name="arm_base" pos="0.5 1.0 0.8">
                    <geom type="cylinder" size="0.05 0.05" rgba="0.1 0.1 0.1 1"/>

                    <body name="shoulder_link" pos="0 0 0.05">
                        <joint name="shoulder_j" type="hinge" axis="0 0 1" limited="false"
                               damping="1.0" armature="0.1"/>
                        <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.03" rgba="0.8 0.2 0.2 1"/>

                        <body name="elbow_link" pos="0.5 0 0">
                            <joint name="elbow_j" type="hinge" axis="0 0 1" limited="true" range="-150 150"
                                   damping="1.0" armature="0.1"/>
                            <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.025" rgba="0.2 0.8 0.2 1"/>

                            <body name="wrist_link" pos="0.4 0 0">
                                <joint name="wrist_j" type="hinge" axis="0 0 1" limited="true" range="-120 120"
                                       damping="0.5" armature="0.05"/>
                                <geom type="box" size="0.03 0.03 0.03" rgba="0.2 0.2 0.8 1"/>

                                <body name="pipette" pos="0.03 0 -0.05">
                                    <geom name="pipette_body" type="cylinder" size="0.01 0.08" rgba="0.9 0.9 0.9 0.7"/>

                                    <!-- Finger 1 -->
                                    <body name="f1_k1" pos="0.01 0 0">
                                        <joint name="f1_j1" type="hinge" axis="0 0 1" limited="true" range="0 90"
                                               damping="0.05" armature="0.01"/>
                                        <geom type="capsule" fromto="0 0 0 0.02 0 0" size="0.005" rgba="0.9 0.1 0.9 1"/>
                                        <body name="f1_k2" pos="0.02 0 0">
                                            <joint name="f1_j2" type="hinge" axis="0 0 1" limited="true" range="0 90"
                                                   damping="0.05" armature="0.01"/>
                                            <geom type="capsule" fromto="0 0 0 0.015 0 0" size="0.004" rgba="0.9 0.1 0.9 1"/>
                                            <body name="f1_k3" pos="0.015 0 0">
                                                <joint name="f1_j3" type="hinge" axis="0 0 1" limited="true" range="0 90"
                                                       damping="0.05" armature="0.01"/>
                                                <geom type="capsule" fromto="0 0 0 0.01 0 0" size="0.003" rgba="0.9 0.1 0.9 1"/>
                                            </body>
                                        </body>
                                    </body>

                                    <!-- Finger 2 -->
                                    <body name="f2_k1" pos="-0.005 0.008 0">
                                        <joint name="f2_j1" type="hinge" axis="0 0 1" limited="true" range="0 90"
                                               damping="0.05" armature="0.01"/>
                                        <geom type="capsule" fromto="0 0 0 -0.01 0.015 0" size="0.005" rgba="0.9 0.1 0.9 1"/>
                                        <body name="f2_k2" pos="-0.01 0.015 0">
                                            <joint name="f2_j2" type="hinge" axis="0 0 1" limited="true" range="0 90"
                                                   damping="0.05" armature="0.01"/>
                                            <geom type="capsule" fromto="0 0 0 -0.008 0.012 0" size="0.004" rgba="0.9 0.1 0.9 1"/>
                                            <body name="f2_k3" pos="-0.008 0.012 0">
                                                <joint name="f2_j3" type="hinge" axis="0 0 1" limited="true" range="0 90"
                                                       damping="0.05" armature="0.01"/>
                                                <geom type="capsule" fromto="0 0 0 -0.005 0.008 0" size="0.003" rgba="0.9 0.1 0.9 1"/>
                                            </body>
                                        </body>
                                    </body>

                                    <!-- Finger 3 -->
                                    <body name="f3_k1" pos="-0.005 -0.008 0">
                                        <joint name="f3_j1" type="hinge" axis="0 0 1" limited="true" range="0 90"
                                               damping="0.05" armature="0.01"/>
                                        <geom type="capsule" fromto="0 0 0 -0.01 -0.015 0" size="0.005" rgba="0.9 0.1 0.9 1"/>
                                        <body name="f3_k2" pos="-0.01 -0.015 0">
                                            <joint name="f3_j2" type="hinge" axis="0 0 1" limited="true" range="0 90"
                                                   damping="0.05" armature="0.01"/>
                                            <geom type="capsule" fromto="0 0 0 -0.008 -0.012 0" size="0.004" rgba="0.9 0.1 0.9 1"/>
                                            <body name="f3_k3" pos="-0.008 -0.012 0">
                                                <joint name="f3_j3" type="hinge" axis="0 0 1" limited="true" range="0 90"
                                                       damping="0.05" armature="0.01"/>
                                                <geom type="capsule" fromto="0 0 0 -0.005 -0.008 0" size="0.003" rgba="0.9 0.1 0.9 1"/>
                                            </body>
                                        </body>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </worldbody>

            <actuator>
                <position joint="shoulder_j" kp="20"  kv="1.0" ctrlrange="-3.14 3.14"/>
                <position joint="elbow_j"    kp="20"  kv="1.0" ctrlrange="-2.6 2.6"/>
                <position joint="wrist_j"    kp="10"  kv="0.5" ctrlrange="-2.1 2.1"/>

                <position joint="f1_j1" kp="0.05" kv="0.005" ctrlrange="0 1.57"/>
                <position joint="f1_j2" kp="0.05" kv="0.005" ctrlrange="0 1.57"/>
                <position joint="f1_j3" kp="0.05" kv="0.005" ctrlrange="0 1.57"/>

                <position joint="f2_j1" kp="0.05" kv="0.005" ctrlrange="0 1.57"/>
                <position joint="f2_j2" kp="0.05" kv="0.005" ctrlrange="0 1.57"/>
                <position joint="f2_j3" kp="0.05" kv="0.005" ctrlrange="0 1.57"/>

                <position joint="f3_j1" kp="0.05" kv="0.005" ctrlrange="0 1.57"/>
                <position joint="f3_j2" kp="0.05" kv="0.005" ctrlrange="0 1.57"/>
                <position joint="f3_j3" kp="0.05" kv="0.005" ctrlrange="0 1.57"/>
            </actuator>
        </mujoco>
        """
        self.model = mujoco.MjModel.from_xml_string(self.xml_model)
        self.data = mujoco.MjData(self.model)

        self.arm_mass = 3.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(70,), dtype=np.float32)

        self.viewer = None
        self.target_transfer_amount = 10.0

        self.pipette_volume = 0.0
        self.beaker_volume = 10.0
        self.petri_volume = 0.0
        self.grasp_force_f1 = 0.0
        self.grasp_force_f2 = 0.0
        self.grasp_force_f3 = 0.0

    def _calculate_motor_resistance(self):
        velocities = self.data.qvel[:12]
        base_resistance = -0.15 * velocities
        fluid_weight_factor = (self.pipette_volume / 10.0) * (self.arm_mass / 6.0) * 0.5
        base_resistance[:3] = base_resistance[:3] - fluid_weight_factor * velocities[:3]
        return base_resistance

    def _get_obs(self):
        beaker_xy = self.data.body("beaker").xpos[:2]
        petri_xy = self.data.geom("petri_geom").xpos[:2]

        arm_positions = []
        arm_velocities = []
        for body_name in ["shoulder_link", "elbow_link", "wrist_link"]:
            arm_positions.extend(self.data.body(body_name).xpos)
            arm_velocities.extend(self.data.body(body_name).cvel[3:])

        hand_angles = self.data.qpos[3:12]
        hand_velocities = self.data.qvel[3:12]

        resistance = self._calculate_motor_resistance()

        fluid_metrics = [
            self.target_transfer_amount,
            self.pipette_volume,
            self.beaker_volume,
            self.petri_volume,
            self.grasp_force_f1,
            self.grasp_force_f2,
            self.grasp_force_f3,
        ]

        obs = np.concatenate([
            beaker_xy,
            petri_xy,
            arm_positions,
            arm_velocities,
            hand_angles,
            hand_velocities,
            resistance,
            fluid_metrics,
        ]).astype(np.float32)

        if len(obs) < 70:
            obs = np.pad(obs, (0, 70 - len(obs)), 'constant')
        return obs[:70]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        beaker_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "beaker")
        petri_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "petri_dish")
        self.model.body_pos[beaker_id] = [
            self.np_random.uniform(0.1, 0.4),
            self.np_random.uniform(0.1, 0.8),
            0.82,
        ]
        self.model.body_pos[petri_id] = [
            self.np_random.uniform(0.6, 0.9),
            self.np_random.uniform(0.1, 0.8),
            0.805,
        ]

        self.beaker_volume = 10.0
        self.pipette_volume = 0.0
        self.petri_volume = 0.0
        self.grasp_force_f1 = 0.0
        self.grasp_force_f2 = 0.0
        self.grasp_force_f3 = 0.0

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        joint_ranges = np.array([
            [-3.14, 3.14],
            [-2.6, 2.6],
            [-2.1, 2.1],
            [0, 1.57], [0, 1.57], [0, 1.57],
            [0, 1.57], [0, 1.57], [0, 1.57],
            [0, 1.57], [0, 1.57], [0, 1.57],
        ])
        targets = np.array([
            lo + (a + 1) / 2 * (hi - lo)
            for a, (lo, hi) in zip(action, joint_ranges)
        ])
        self.data.ctrl[:] = targets
        mujoco.mj_step(self.model, self.data)

        pipette_tip = self.data.geom("pipette_body").xpos
        beaker_pos = self.data.body("beaker").xpos
        dish_pos = self.data.geom("petri_geom").xpos

        dist_to_beaker = np.linalg.norm(pipette_tip - beaker_pos)
        dist_to_dish = np.linalg.norm(pipette_tip - dish_pos)

        self.grasp_force_f1 = float(np.sum(self.data.qpos[3:6]))
        self.grasp_force_f2 = float(np.sum(self.data.qpos[6:9]))
        self.grasp_force_f3 = float(np.sum(self.data.qpos[9:12]))

        reward = 0.0
        terminated = False
        truncated = False

        if self.pipette_volume < self.target_transfer_amount and self.beaker_volume > 0:
            reward -= dist_to_beaker * 5.0
            if dist_to_beaker < 0.06:
                if 0.2 < self.grasp_force_f1 < 1.2:
                    transfer = min(0.1, self.beaker_volume)
                    self.pipette_volume += transfer
                    self.beaker_volume -= transfer
                    reward += 15.0
                else:
                    reward -= 10.0
        else:
            reward -= dist_to_dish * 5.0
            if dist_to_dish < 0.07:
                if 0.2 < self.grasp_force_f1 < 1.2:
                    transfer = min(0.1, self.pipette_volume)
                    self.petri_volume += transfer
                    self.pipette_volume -= transfer
                    reward += 20.0
                    if self.petri_volume >= self.target_transfer_amount:
                        reward += 500.0
                        terminated = True

        return self._get_obs(), reward, terminated, truncated, {"delivered": self.petri_volume}

    def render(self):
        if self.render_mode == "human" and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    env = AdvancedLiquidTransferEnv()
    obs, _ = env.reset()
    print(f"Environment compiled. Observation shape: {obs.shape}")
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.01)
        if terminated or truncated:
            break
    env.close()