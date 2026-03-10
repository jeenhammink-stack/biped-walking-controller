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

import matplotlib.pyplot as plt

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
    damping_task: mink.DampingTask,
    com_target: np.ndarray,
    stance_des_pos: np.ndarray,
    swing_des_pos: np.ndarray,
    backpack_angle: float,
    params: InvKinSolverParamsMujoco,
):

    tasks = [com_task, stance_foot_task, swing_foot_task, backpack_orientation_task, damping_task]
    constraints = []
    # Set targets
    stance_foot_rotation = SO3.from_x_radians(0) @ SO3.from_y_radians(np.deg2rad(0))   # no rotation
    stance_foot_target = SE3.from_translation(stance_des_pos) @ SE3.from_rotation(stance_foot_rotation)
    stance_foot_task.set_target(stance_foot_target)
    constraints.append(stance_foot_task)

    swing_foot_rotation = SO3.from_x_radians(0) @ SO3.from_y_radians(np.deg2rad(0))   # no rotation
    swing_foot_target = SE3.from_translation(swing_des_pos) @ SE3.from_rotation(swing_foot_rotation)
    swing_foot_task.set_target(swing_foot_target)

    com_task.set_target(com_target)

    rotation_backpack = SO3.from_x_radians(0) @ SO3.from_y_radians(np.deg2rad(backpack_angle))   # Stay upright with slight forward tilt.
    target_backpack = SE3.from_rotation(rotation_backpack)  
    backpack_orientation_task.set_target(target_backpack)

    vel = mink.solve_ik(
        configuration, tasks, params.dt, params.solver, damping=params.damping, constraints=constraints
    )
    return configuration.integrate(vel, params.dt)


if __name__ == "__main__":
    print("Running inverse kinematic mujoco script")
    exo = Exo()
    model = exo.mj_model_air
    q_init = model.keyframe("home").qpos.copy()
    configuration = mink.Configuration(model, q_init)

    tasks = [
        backpack_orientation_task := mink.FrameTask(
            frame_name="imu_backpack",
            frame_type="site",
            position_cost=0.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        ),
        com_task := mink.ComTask(cost=10.0),
        left_foot_task := mink.FrameTask(
                frame_name="pos_L_foot",
                frame_type="site",
                position_cost=0.0,
                orientation_cost=0.0,
                lm_damping=1.0,
            ),
        right_foot_task := mink.FrameTask(
                frame_name="pos_R_foot",
                frame_type="site",
                position_cost=0.0,
                orientation_cost=0.0,
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
        test_trajectory = np.linspace(0, 2, 500)
        swing_init_pos = data.site_xpos[left_foot_site_id].copy()
        print(f"Initial stance foot position: {stance_des_pos}") 
        target_backpack = 0.0
        backpack_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "backpack")
        # Get the subtree COM for the backpack body (includes all child bodies)
        current_com = data.subtree_com[backpack_body_id].copy()
        print(f"Current backpack COM: {current_com}")
        counter = 0
        data.qpos[:] = model.keyframe("home").qpos.copy()  # Start from the "home" keyframe pose.
        mujoco.mj_forward(model, data)  # Update all dependent quantities
        configuration.update(data.qpos)  # Sync configuration with initial pose
        buffer_length = 100
        i = 0
        sin = 0.2*np.array([np.sin(2 * np.pi * 0.5 * t) for t in test_trajectory])-0.15  # 0.5 Hz sine wave for foot trajectory
        com_pos = np.zeros((500,3))
        com_ref = np.zeros((500,3))
        fe_pos = np.zeros(500)
        fe_ref = np.zeros(500)
        while viewer.is_running():
            while i < buffer_length:
                start_time = time.time()
                mujoco.mj_forward(model, data)
                elapsed = time.time() - start_time
                if elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)
                viewer.sync()
                i += 1
            start_time = time.time()
            # print("Solving inverse kinematics...")
            if counter < len(test_trajectory)-1:
                # swing_des_pos = swing_init_pos + np.array([0, 0.0, test_trajectory[counter]/2])  # Move swing foot forward and slightly up.
                com_target = current_com + np.array([test_trajectory[counter], test_trajectory[counter]/2, 0])
                com_ref[counter] = com_target
                com_pos[counter] = data.subtree_com[backpack_body_id].copy() 
            # q_des = solve_inv_kinematics_mujoco(
            #     configuration=configuration,
            #     com_task=com_task,
            #     stance_foot_task=right_foot_task,
            #     swing_foot_task=left_foot_task,
            #     backpack_orientation_task=backpack_orientation_task,
            #     com_target=com_target,
            #     stance_des_pos=stance_des_pos,
            #     swing_des_pos=swing_init_pos,
            #     backpack_angle=target_backpack,
            #     params=params,
            # )
            # data.qpos[:] = q_des
                data.ctrl[1] = sin[counter]  # Oscillate the FE joint of the left leg to visualize motion.
                fe_ref[counter] = sin[counter]
                fe_pos[counter] = data.qpos[1]
            mujoco.mj_step(model, data)
            # mujoco.mj_forward(model, data)  # Update all dependent quantities after changing qpos
            elapsed = time.time() - start_time
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)
            viewer.sync()
            counter += 1
            print(f"End of iteration {counter}")
        plt.plot(fe_ref, label="FE joint reference")
        plt.plot(fe_pos, label="FE joint actual")
        plt.legend()
        plt.show()
