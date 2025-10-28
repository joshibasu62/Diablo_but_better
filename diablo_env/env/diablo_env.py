import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os
import time

from diablo_env.env.utils import load_robot, set_gravity, step_simulation


class DiabloEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode="human"):
        super().__init__()

        # Simulation setup
        self.render_mode = render_mode
        self.physics_client = None

        # Action space: let's assume 8 joints (4 per leg)
        self.num_joints = 8
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )

        # Observation space (example: joint positions + base position)
        obs_dim = self.num_joints + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Paths
        self.assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
        self.urdf_path = os.path.join(self.assets_dir, "urdf", "robot.urdf")

        self.robot_id = None

    def _connect(self):
        if self.render_mode == "human":
            return p.connect(p.GUI)
        else:
            return p.connect(p.DIRECT)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.physics_client is not None:
            p.disconnect(self.physics_client)

        self.physics_client = self._connect()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.loadURDF("plane.urdf")
        set_gravity()

        self.robot_id = load_robot(self.urdf_path, base_position=[0, 0, 0.2])

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        joint_states = [p.getJointState(self.robot_id, i)[0] for i in range(self.num_joints)]
        obs = np.array(list(pos) + joint_states, dtype=np.float32)
        return obs

    def step(self, action):
        """
        Step function using safer PD velocity control.
        Action: desired joint velocities [-1, 1] per joint.
        """
        max_vel = 1.0      # rad/s
        max_torque = 5.0   # Nm per joint

        # Clip actions to safe velocity range
        action = np.clip(action, -max_vel, max_vel)

        for i in range(self.num_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=i,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=action[i],
                force=max_torque
            )

        # Step the simulation
        p.stepSimulation()

        # Get observation
        obs = self._get_obs()

        # Reward: upright + small forward motion bonus
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        _, ang_vel = p.getBaseVelocity(self.robot_id)

        upright_reward = max(0, pos[2])            # reward for height
        stability_penalty = np.sum(np.square(ang_vel))  # penalize spinning
        reward = upright_reward - 0.1 * stability_penalty

        # Done if robot tips or falls
        euler = p.getEulerFromQuaternion(orn)
        done = pos[2] < 0.15 or abs(euler[0]) > 0.5 or abs(euler[1]) > 0.5

        return obs, reward, done, False, {}



    def render(self):
        if self.render_mode == "rgb_array":
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0.2],
                distance=1.5,
                yaw=50,
                pitch=-35,
                roll=0,
                upAxisIndex=2,
            )
            proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=100.0)
            (_, _, px, _, _) = p.getCameraImage(
                width=640, height=480, viewMatrix=view_matrix, projectionMatrix=proj_matrix
            )
            rgb_array = np.array(px)
            return rgb_array

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
