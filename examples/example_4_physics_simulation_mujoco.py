import math
import argparse
import typing
from dataclasses import dataclass
from pathlib import Path

import mink
import mujoco
import numpy as np
import pybullet as pb
import pinocchio as pin
from matplotlib import pyplot as plt

from biped_walking_controller import model
from biped_walking_controller.foot import (
    compute_feet_trajectories,
    BezierCurveFootPathGenerator,
    compute_steps_sequence,
)

from biped_walking_controller.inverse_kinematic import InvKinSolverParams, solve_inverse_kinematics
from biped_walking_controller.inverse_kinematic_mujoco import solve_inv_kinematics_mujoco, InvKinSolverParamsMujoco
from biped_walking_controller.model_mujoco import Exo
from biped_walking_controller.plot import plot_feet_and_com, plot_contact_forces

from biped_walking_controller.preview_control import (
    PreviewControllerParams,
    compute_preview_control_matrices,
    update_control,
    compute_zmp_ref,
    cubic_spline_interpolation,
)

from biped_walking_controller.model import Talos, q_from_base_and_joints

from biped_walking_controller.simulation import (
    _snap_feet_to_plane,
    _compute_base_from_foot_target,
    Simulator,
)


@dataclass
class GeneralParams:
    dt = 1.0 / 240.0
    t_ss = 0.8  # Single support phase time window
    t_ds = 0.3  # Double support phase time window
    t_init = 2.0  # Initialization phase (transition from still position to first step)
    t_end = 0.4
    n_steps = 15  # Number of steps executed by the robot
    l_stride = 0.1  # Length of the stride
    max_height_foot = 0.01  # Maximal height of the swing foot
    t_preview = 1.6
    n_solver_iter = 200


def get_standard_params() -> typing.Tuple[GeneralParams, PreviewControllerParams]:
    general_params = GeneralParams()

    ctrler_params = PreviewControllerParams(
        zc=0.89,
        g=9.81,
        Qe=1.0,
        Qx=np.zeros((3, 3)),
        R=1e-6,
        n_preview_steps=int(round(general_params.t_preview / general_params.dt)),
    )

    return general_params, ctrler_params


def get_accurate_sim_params() -> typing.Tuple[GeneralParams, PreviewControllerParams]:
    general_params, ctrler_params = get_standard_params()

    # Specific params
    general_params.dt = 1.0 / 1000.0
    general_params.t_ss = 0.6
    general_params.t_ds = 0.1
    general_params.n_steps = 15
    general_params.l_stride = 0.15
    general_params.max_height_foot = 0.02
    general_params.n_solver_iter = 1500

    return general_params, ctrler_params


def main():
    scen_params, ctrler_params = get_accurate_sim_params()
    
    model = Exo().mj_model
    data = mujoco.MjData(model)
    ##############
    # MINK setup #
    ##############
    configuration = mink.Configuration(model)
    left_foot_name = "pos_L_foot"
    right_foot_name = "pos_R_foot"
    backpack_name = "imu_backpack"
    tasks = [
        backpack_orientation_task := mink.FrameTask(
            frame_name=backpack_name,
            frame_type="site",
            position_cost=0.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        com_task := mink.ComTask(cost=10.0),
        left_foot_task := mink.FrameTask(
                frame_name=left_foot_name,
                frame_type="site",
                position_cost=10.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            ),
        right_foot_task := mink.FrameTask(
                frame_name=right_foot_name,
                frame_type="site",
                position_cost=10.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            )]
    solver = "daqp"
    dt = 0.01
    ik_solve_params = InvKinSolverParamsMujoco(
        fixed_foot=left_foot_name,
        swing_foot=right_foot_name
    )

    ##############
    # LIPM setup #
    ##############
    right_foot_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pos_R_foot")
    left_foot_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pos_L_foot")
    backpack_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu_backpack")
    # Get site position (3D world coordinates)
    right_foot_site_pos = data.site_xpos[right_foot_site_id].copy()
    left_foot_site_pos = data.site_xpos[left_foot_site_id].copy()
    backpack_site_pos = data.site_xpos[backpack_site_id].copy()

    feet_mid = 0.5 * (right_foot_site_pos + left_foot_site_pos)
    com_initial_target = np.array([feet_mid[0], feet_mid[1], ctrler_params.zc])  # Initial CoM target above the midpoint of the feet.

    ctrler_mat = compute_preview_control_matrices(ctrler_params, scen_params.dt)

    # Run a single iteration of inverse kinemetics to get desired initial position of exo
    solve_inv_kinematics_mujoco(configuration=configuration, 
                                tasks=tasks, 
                                com_target=com_initial_target, 
                                stance_des_pos=left_foot_site_pos, 
                                swing_des_pos=right_foot_site_pos, 
                                backpack_target=backpack_site_pos, 
                                params=ik_solve_params,
                                com_task=com_task,
                                left_foot_task=left_foot_task,
                                right_foot_task=right_foot_task,
                                backpack_orientation_task=backpack_orientation_task)
    
if __name__ == "__main__":
    main()