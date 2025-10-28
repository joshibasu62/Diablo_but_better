import pybullet as p
import pybullet_data
import os

def load_robot(urdf_path, base_position=[0, 0, 0.3]):
    """Loads a robot URDF into PyBullet"""
    robot_id = p.loadURDF(urdf_path, basePosition=base_position, useFixedBase=False)
    return robot_id

def set_gravity(gravity=-9.81):
    """Sets gravity in the world"""
    p.setGravity(0, 0, gravity)

def step_simulation():
    """Steps the simulation once"""
    p.stepSimulation()
