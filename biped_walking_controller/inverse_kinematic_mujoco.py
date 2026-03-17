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
from biped_walking_controller.controller_position import low_level_update
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
    stance_des_quat: np.ndarray,
    swing_des_pos: np.ndarray,
    swing_des_quat: np.ndarray,
    backpack_angle: float,
    params: InvKinSolverParamsMujoco,
):

    tasks = [com_task, stance_foot_task, swing_foot_task, backpack_orientation_task, damping_task]
    constraints = []
    # Set targets
    stance_foot_rotation = SO3(wxyz=stance_des_quat)  # no rotation
    stance_foot_target = SE3.from_translation(stance_des_pos) @ SE3.from_rotation(stance_foot_rotation)
    stance_foot_task.set_target(stance_foot_target)
    constraints.append(stance_foot_task)

    swing_foot_rotation = SO3(wxyz=swing_des_quat)  # no rotation
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
    model = exo.mj_model_torque
    data = mujoco.MjData(model)
    data.qpos[:] = model.keyframe("home").qpos.copy()
    mujoco.mj_forward(model, data)  # Update all dependent quantities after changing qpos
    # q_init = model.keyframe("home").qpos.copy()
    # configuration = mink.Configuration(model, q_init)

    # tasks = [
    #     backpack_orientation_task := mink.FrameTask(
    #         frame_name="imu_backpack",
    #         frame_type="site",
    #         position_cost=0.0,
    #         orientation_cost=0.0,
    #         lm_damping=1.0,
    #     ),
    #     com_task := mink.ComTask(cost=10.0),
    #     left_foot_task := mink.FrameTask(
    #             frame_name="pos_L_foot",
    #             frame_type="site",
    #             position_cost=0.0,
    #             orientation_cost=0.0,
    #             lm_damping=1.0,
    #         ),
    #     right_foot_task := mink.FrameTask(
    #             frame_name="pos_R_foot",
    #             frame_type="site",
    #             position_cost=0.0,
    #             orientation_cost=0.0,
    #             lm_damping=1.0,
    #         )]
    # solver = "daqp"
    # dt = 0.01
    # model = configuration.model
    # data = configuration.data
    # mujoco.mj_forward(model, data)
    # params = InvKinSolverParamsMujoco(fixed_foot="pos_R_foot", swing_foot="pos_L_foot")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # print("Viewer launched successfully.")
        # mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        # print("Camera initialized.")

        # right_foot_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pos_R_foot")
        # left_foot_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pos_L_foot")
        # mujoco.mj_forward(model, data)
        # stance_des_pos = data.site_xpos[right_foot_site_id].copy()
        # test_trajectory = np.linspace(0, 2, 500)
        # swing_init_pos = data.site_xpos[left_foot_site_id].copy()
        # print(f"Initial stance foot position: {stance_des_pos}") 
        # target_backpack = 0.0
        # backpack_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "backpack")
        # # Get the subtree COM for the backpack body (includes all child bodies)
        index = 0
        reference = model.key_qpos
        data.qpos[:] = reference[0].copy()
        mujoco.mj_forward(model, data)
        joint_gains = {
            "LEFT_ANKLE_DPF":  {"kp": 75,       "kd": 2.5},
            "LEFT_ANKLE_IE":   {"kp": 60,       "kd": 0.8},
            "LEFT_HIP_AA":     {"kp": 100,       "kd": 2.5},
            "LEFT_HIP_FE":     {"kp": 100,       "kd": 0.8},
            "LEFT_KNEE":       {"kp": 40,       "kd": 0.8},
            "RIGHT_ANKLE_DPF": {"kp": 75,       "kd": 2.5},
            "RIGHT_ANKLE_IE":  {"kp": 60,       "kd": 0.8},
            "RIGHT_HIP_AA":    {"kp": 100,       "kd": 2.5},
            "RIGHT_HIP_FE":    {"kp": 100,       "kd": 0.8},  # 120, 0, 0.8
            "RIGHT_KNEE":      {"kp": 40,       "kd": 0.8},
            
        }
        while viewer.is_running():
            start_time = time.time()

            low_level_update(model=model,data=data, joint_gains=joint_gains, desired_pos=reference[index, 7:])

            mujoco.mj_step(model, data)
            # mujoco.mj_forward(model, data)  # Update all dependent quantities after changing qpos
            elapsed = time.time() - start_time
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)
            viewer.sync()
            index +=1
            index = index % len(reference)
            print(f"time: {data.time:.3f}")

