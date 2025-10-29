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

        self.torque_left_joint_1 = 10.0
        self.torque_left_joint_2 = 8.0
        # self.torque_left_joint_3 = 6.0   it is knee so motor not required     
        self.torque_left_joint_4 = 10.0

        self.torque_right_joint_1 = 10.0
        self.torque_right_joint_2 = 8.0     
        # self.torque_right_joint_3 = 6.0     it is also knee   
        self.torque_right_joint_4 = 10.0

        self.torques = [
            self.torque_left_joint_1, #body joint left
            self.torque_left_joint_2, #hip joint left
            # self.torque_left_joint_3,
            self.torque_left_joint_4, #wheel joint left
            self.torque_right_joint_1,
            self.torque_right_joint_2,
            # self.torque_right_joint_3,
            self.torque_right_joint_4,
            ]

        

        # Simulation setup
        self.render_mode = render_mode

        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        set_gravity()

        
        # Paths
        self.assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
        self.urdf_path = os.path.join(self.assets_dir, "urdf", "robot.urdf")

        self.robot_id = None

        # Load 
        p.loadURDF("plane.urdf",basePosition=[0, 0, 0])

        self.robot_id = load_robot(self.urdf_path, base_position=[0, 0, 0.495])


        # Print joint info for debugging will remove later after testing
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            print(i, info[1].decode("utf-8"), info[2])


        # Get all revolute joints
        self.revolute_joints = []
        self.num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_type = info[2]
            if joint_type == p.JOINT_REVOLUTE:  # Only revolute joints
                self.revolute_joints.append(i)
        self.num_rev_joints = len(self.revolute_joints)

        # Remove this print after testing
        print("Revolute joints:", self.revolute_joints)

        # Example: define action space based on revolute joints
        self.num_joints_with_input = len(self.torques)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints_with_input,), dtype=np.float32
        )

        # Example: observation space = joint positions + velocities
        obs_dim = 2 * self.num_rev_joints + 3*2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def get_observation(self):
        joints = p.getJointStates(self.robot_id, self.revolute_joints)
        positions = [state[0] for state in joints]
        velocities = [state[1] for state in joints]
        base_pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        return np.array(positions + velocities + list(base_pos) + list(orn), dtype=np.float32)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.robot_id is not None:
            p.removeBody(self.robot_id)

        # Load robot 
        self.robot_id = load_robot(self.urdf_path, base_position=[0, 0, 0.495])

        obs = self.get_observation()
        return obs, {}


    def step(self, action):
        #position control
        max_force = 10.0
        position_gain = 0.5
        
        # Convert actions to small position changes around neutral position
        neutral_positions = [0.0] * self.num_joints  # Adjust if your robot has different neutral pose
        target_positions = neutral_positions + action * 0.1  # Small position changes
        
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_positions[i],
                force=max_force,
                positionGain=position_gain
            )

        # Step the simulation
        p.stepSimulation()

        # Get observation
        obs = self.get_observation()

        # Reward: upright + small forward motion bonus
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        _, ang_vel = p.getBaseVelocity(self.robot_id)

        upright_reward = max(0, pos[2])            # reward for height
        stability_penalty = np.sum(np.square(ang_vel))  # penalize spinning
        reward = upright_reward - 0.1 * stability_penalty

        # Done if robot tips or falls
        euler = p.getEulerFromQuaternion(orn)
        done = pos[2] < 0.05 or abs(euler[0]) > 0.5 or abs(euler[1]) > 0.5

        truncated = False
        terminated = done
        info = {}

        return obs, reward, truncated, terminated, info



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
