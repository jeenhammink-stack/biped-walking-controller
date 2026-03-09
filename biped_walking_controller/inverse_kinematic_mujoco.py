import string
import time
import typing
from dataclasses import dataclass
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

# import pinocchio as pin
from qpsolvers import solve_qp
import mink
from mink import SE3, SO3
from biped_walking_controller.model_mujoco import Exo



@dataclass
class InvKinSolverParamsMujoco:
    """
    Parameters for inverse-kinematics QP with CoM, feet, and torso tasks.

    Attributes
    ----------
    fixed_foot : str
        Name of the stance foot site (hard equality).
    swing_foot : str
        Name of the swing foot site (soft task).
    damping : float
       Levenberg-Marquardt damping applied to all tasks. Higher values improve numerical stability but slow down task convergence. 
       This value applies to all dofs, including floating-base coordinates.
    dt: float 
    Integration timestep in [s].
    solver: str
    Backend quadratic programming (QP) solver. (use daqp)
    """
    fixed_foot : str = None
    swing_foot : str = None
    backpack : str = "imu_backpack"
    damping : float = 1e-1
    dt: float = 0.01
    solver: str = "daqp"

def solve_inv_kinematics_mujoco(
    configuration: mink.Configuration,
    com_task: mink.ComTask,
    stance_foot_task: mink.FrameTask,
    swing_foot_task: mink.FrameTask,
    backpack_orientation_task: mink.FrameTask,
    com_target: np.ndarray,
    stance_des_pos: np.ndarray,
    swing_des_pos: np.ndarray,
    backpack_angle: float,
    params: InvKinSolverParamsMujoco,
):
    # print("Setting up inverse kinematics problem...")
    # Set targets
    tasks = [com_task, stance_foot_task, swing_foot_task, backpack_orientation_task]
    constraints = []
    stance_foot_rotation = SO3.from_x_radians(0) @ SO3.from_y_radians(np.deg2rad(0))   # no rotation
    stance_foot_target = SE3.from_translation(stance_des_pos) @ SE3.from_rotation(stance_foot_rotation)
    stance_foot_task.set_target(stance_foot_target)
    constraints.append(stance_foot_task)

    swing_foot_rotation = SO3.from_x_radians(0) @ SO3.from_y_radians(np.deg2rad(0))   # no rotation
    swing_foot_target = SE3.from_translation(swing_des_pos) @ SE3.from_rotation(swing_foot_rotation)
    swing_foot_task.set_target(swing_foot_target)

    com_task.set_target(com_target)
    backpack_pose = configuration.get_transform_frame_to_world("imu_backpack", "site")
    rotation_backpack = SO3.from_x_radians(0) @ SO3.from_y_radians(np.deg2rad(backpack_angle))   # Stay upright with slight forward tilt.
    target_backpack = SE3.from_rotation(rotation_backpack)  
    backpack_orientation_task.set_target(target_backpack)
    # print("Solving QP...")      
    vel = mink.solve_ik(
        configuration, tasks, params.dt, params.solver, damping=params.damping, constraints=constraints
    )
    return configuration.integrate(vel, params.dt)


if __name__ == "__main__":
    print("Running inverse kinematic mujoco script")
    exo = Exo()
    model = exo.mj_model

    configuration = mink.Configuration(model)

    tasks = [
        backpack_orientation_task := mink.FrameTask(
            frame_name="imu_backpack",
            frame_type="site",
            position_cost=0.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        com_task := mink.ComTask(cost=10.0),
        left_foot_task := mink.FrameTask(
                frame_name="pos_L_foot",
                frame_type="site",
                position_cost=10.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            ),
        right_foot_task := mink.FrameTask(
                frame_name="pos_R_foot",
                frame_type="site",
                position_cost=10.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            )]
    solver = "daqp"
    dt = 0.01
    model = configuration.model
    data = configuration.data
    mujoco.mj_forward(model, data)
    params = InvKinSolverParamsMujoco(fixed_foot="pos_R_foot", swing_foot="pos_L_foot")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer launched successfully.")
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        print("Camera initialized.")
        # backpack_pose = configuration.get_transform_frame_to_world("imu_backpack", "site")
        # rotation_backpack = SO3.from_x_radians(0) @ SO3.from_y_radians(np.deg2rad(3))   # Stay upright with slight forward tilt.
        # target_backpack = SE3.from_rotation(rotation_backpack)  

        # left_foot_pose = configuration.get_transform_frame_to_world("pos_L_foot", "site")
        # left_foot_translation = np.array([0.2, 0, 0.1])
        # left_foot_rotation = SO3.from_x_radians(0) @ SO3.from_y_radians(np.deg2rad(0))   # no rotation
        # left_foot_target = SE3.from_translation(left_foot_translation) @ left_foot_pose  # Translation in world frame.
        # left_foot_target = left_foot_target @ SE3.from_rotation(left_foot_rotation)
        # swing_target = left_foot_target

        # right_foot_pose = configuration.get_transform_frame_to_world("pos_R_foot", "site")
        # right_foot_translation = np.array([0.0, 0, 0.0])
        # right_foot_rotation = SO3.from_x_radians(0) @ SO3.from_y_radians(np.deg2rad(0))   # no rotation
        # right_foot_target = SE3.from_translation(right_foot_translation) @ right_foot_pose  # Translation in world frame.
        # right_foot_target = right_foot_target @ SE3.from_rotation(right_foot_rotation)
        # stance_target = right_foot_target
        right_foot_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pos_R_foot")
        left_foot_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pos_L_foot")
        mujoco.mj_forward(model, data)
        stance_des_pos = data.site_xpos[right_foot_site_id].copy()
        test_trajectory = np.linspace(0, 0.3, 100)
        swing_init_pos = data.site_xpos[left_foot_site_id].copy()
        print(f"Initial stance foot position: {stance_des_pos}") 
        target_backpack = 0.0
        backpack_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "backpack")
        # Get the subtree COM for the backpack body (includes all child bodies)
        current_com = data.subtree_com[backpack_body_id].copy()
        print(f"Current backpack COM: {current_com}")
        com_target = current_com + np.array([0.3, 0.0, 0.0])  # Move CoM by 10 cm from current position.
        counter = 0
        while viewer.is_running():
            start_time = time.time()
            # print("Solving inverse kinematics...")
            if counter < len(test_trajectory):
                swing_des_pos = swing_init_pos + np.array([test_trajectory[counter], 0.0, 0.0])
            qpos = solve_inv_kinematics_mujoco(
                configuration=configuration,
                com_task=com_task,
                stance_foot_task=right_foot_task,
                swing_foot_task=left_foot_task,
                backpack_orientation_task=backpack_orientation_task,
                com_target=com_target,
                stance_des_pos=stance_des_pos,
                swing_des_pos=swing_des_pos,
                backpack_angle=target_backpack,
                params=params,
            )
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            elapsed = time.time() - start_time
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)
            viewer.sync()
            counter += 1
