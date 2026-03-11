import math
import argparse
from time import time
import time
import typing
from dataclasses import dataclass
from pathlib import Path

import mink
import mujoco
import mujoco.viewer
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




@dataclass
class GeneralParams:
    dt = 1.0 / 240.0
    t_ss = 0.8  # Single support phase time window
    t_ds = 0.3  # Double support phase time window
    t_init = 2.0  # Initialization phase (transition from still position to first step)
    t_end = 0.4
    n_steps = 15  # Number of steps executed by the robot
    l_stride = 0.1  # Length of the stride
    max_height_foot = 0.30  # Maximal height of the swing foot
    t_preview = 1.6
    n_solver_iter = 200


def get_standard_params() -> typing.Tuple[GeneralParams, PreviewControllerParams]:
    general_params = GeneralParams()

    ctrler_params = PreviewControllerParams(
        zc=0.55, # Desired exo COM height
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
    general_params.dt = 1.0 / 200.0  # MuJoCo default timestep
    general_params.t_ss = 0.8 # Longer ss phase
    general_params.t_ds = 0.1
    general_params.n_steps = 5
    general_params.l_stride = 0.15
    general_params.max_height_foot = 0.03 # Lower foot height
    general_params.n_solver_iter = 1500

    ctrler_params.n_preview_steps = int(round(general_params.t_preview / general_params.dt))
    return general_params, ctrler_params


def main():
    scen_params, ctrler_params = get_accurate_sim_params()
    
    model = Exo().mj_model
    configuration = mink.Configuration(model)
    data = configuration.data
    data.qpos[:] = model.keyframe("home").qpos.copy()

    data.ctrl[:] = model.keyframe("home").qpos[7:].copy()  # Set initial control inputs to match the home pose
    mujoco.mj_step(model, data)


    ##############
    # MINK setup #
    ##############

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
        com_task := mink.ComTask(cost=[10,10,10]),
        left_foot_task := mink.FrameTask(
                frame_name=left_foot_name,
                frame_type="site",
                position_cost=30.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            ),
        right_foot_task := mink.FrameTask(
                frame_name=right_foot_name,
                frame_type="site",
                position_cost=30.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            ),
        damping_task := mink.DampingTask(model=model, cost=2)
        ]
    ik_sol_params = InvKinSolverParamsMujoco(
        fixed_foot=left_foot_name,
        swing_foot=right_foot_name,
        dt=0.01,
    )

    ##############
    # LIPM setup #
    ##############
    right_foot_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pos_R_foot")
    left_foot_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pos_L_foot")
    backpack_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu_backpack")
    # Get site position (3D world coordinates)
    for i in range(10):
        mujoco.mj_step(model, data)
        right_foot_target = data.site_xpos[right_foot_site_id].copy()
        print(f"Initial right foot target: {right_foot_target}")
        # rf_target[2] = 0.0  # Ensure initial foot height is zero for the LIPM.
        left_foot_target = data.site_xpos[left_foot_site_id].copy()
    print(f"Initial left foot target: {left_foot_target}")
    # lf_target[2] = 0.0  # Ensure initial foot height is zero for the LIPM.
    backpack_site_pos = data.site_xpos[backpack_site_id].copy()
    backpack_angle = 3.0  # degrees

    feet_mid = 0.5 * (right_foot_target + left_foot_target)
    des_height = 0.59
    com_initial_target = np.array([feet_mid[0]+0.05, feet_mid[1], des_height])  # Initial CoM target above the midpoint of the feet.
    backpack_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "backpack")
    # Get the subtree COM for the backpack body (includes all child bodies)

    com_initial = data.subtree_com[backpack_body_id].copy()
    print(f"Initial CoM position from MuJoCo: {com_initial}, Initial backpack position: {backpack_site_pos}")
    # com_initial_target[2] = ctrler_params.zc + 0.05 # Set initial CoM height to desired value for the LIPM.
    ctrler_mat = compute_preview_control_matrices(ctrler_params, scen_params.dt)

    steps_pose, steps_ids = compute_steps_sequence(
        rf_initial_pose=right_foot_target,
        lf_initial_pose=left_foot_target,
        n_steps=scen_params.n_steps,
        l_stride=scen_params.l_stride,
    )


    t, left_foot_path, right_foot_path, phases = compute_feet_trajectories(
        rf_initial_pose=right_foot_target,
        lf_initial_pose=left_foot_target,
        n_steps=scen_params.n_steps,
        steps_pose=steps_pose,
        t_ss=scen_params.t_ss,
        t_ds=scen_params.t_ds,
        t_init=scen_params.t_init,
        t_final=scen_params.t_end,
        dt=scen_params.dt,
        traj_generator=BezierCurveFootPathGenerator(scen_params.max_height_foot),
    )
    print(f"left_foot_path.shape: {left_foot_path.shape}, right_foot_path.shape: {right_foot_path.shape}, phases.shape: {phases.shape}")
    zmp_ref = compute_zmp_ref(
        t=t,
        com_initial_pose=com_initial_target[0:2],
        steps=steps_pose[:, 0:2],
        ss_t=scen_params.t_ss,
        ds_t=scen_params.t_ds,
        t_init=scen_params.t_init,
        t_final=scen_params.t_end,
        interp_fn=cubic_spline_interpolation,
    )

    zmp_padded = np.vstack(
        [zmp_ref, np.repeat(zmp_ref[-1][None, :], ctrler_params.n_preview_steps, axis=0)]
    )
    zmp_pos = np.zeros((len(phases), 3))

    x_k = np.array([0.0, com_initial_target[0], 0.0, 0.0], dtype=float)
    y_k = np.array([0.0, com_initial_target[1], 0.0, 0.0], dtype=float)

    com_pin_pos = np.zeros((len(phases), 3))
    com_ref_pos = np.zeros((len(phases), 3))
    com_pb_pos = np.zeros((len(phases), 3))

    lf_ref_pos = np.zeros((len(phases), 3))
    lf_pin_pos = np.zeros((len(phases), 3))
    lf_pb_pos = np.zeros((len(phases), 3))

    rf_ref_pos = np.zeros((len(phases), 3))
    rf_pin_pos = np.zeros((len(phases), 3))
    rf_pb_pos = np.zeros((len(phases), 3))

    zmp_pos = np.zeros((len(phases), 3))

    rf_forces = np.zeros((len(phases), 1))
    lf_forces = np.zeros((len(phases), 1))

    qpos_des = np.zeros((len(phases), model.nq))
    qpos_act = np.zeros((len(phases), model.nq))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer launched successfully.")
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        for k, _ in enumerate(phases[:-2]):
            start_time = time.time()
            zmp_ref_horizon = zmp_padded[k + 1 : k + ctrler_params.n_preview_steps]

            _, x_k, y_k = update_control(
                ctrler_mat, zmp_padded[k], zmp_ref_horizon, x_k.copy(), y_k.copy()
            )

            # zmp_pos[k] = calculate_zmp(model, data) (optional for plotting purposes)
            # com_target = np.array([x_k[1], y_k[1], ctrler_params.zc])
            com_target = np.array([x_k[1], y_k[1], des_height])  # Keep the initial CoM height constant for the LIPM.

            if phases[k] < 0.0:
                # Set right foot as stance and left foot as swing
                ik_sol_params.fixed_foot_frame = right_foot_name
                ik_sol_params.moving_foot_frame = left_foot_name

                q_des = solve_inv_kinematics_mujoco(
                    configuration=configuration, 
                    com_task=com_task,
                    stance_foot_task=right_foot_task,
                    swing_foot_task=left_foot_task,
                    backpack_orientation_task=backpack_orientation_task,
                    damping_task=damping_task,
                    com_target=com_target,
                    stance_des_pos=right_foot_target,
                    swing_des_pos=left_foot_path[k],
                    backpack_angle=backpack_angle,
                    params=ik_sol_params,
                )
                
                if phases[k + 1] > 0.0:
                # Update the stance target so that u dont use the value of one gait cycle before
                    left_foot_target = left_foot_path[k + 1].copy()
            else:
                ik_sol_params.fixed_foot_frame = left_foot_name
                ik_sol_params.moving_foot_frame = right_foot_name

                q_des = solve_inv_kinematics_mujoco(
                    configuration=configuration, 
                    com_task=com_task,
                    stance_foot_task=left_foot_task,
                    swing_foot_task=right_foot_task,
                    backpack_orientation_task=backpack_orientation_task,
                    damping_task=damping_task,
                    com_target=com_target,
                    stance_des_pos=left_foot_target,
                    swing_des_pos=right_foot_path[k],
                    backpack_angle=backpack_angle,
                    params=ik_sol_params,
                )

                if phases[k + 1] < 0.0:
                    right_foot_target = right_foot_path[k + 1].copy()
            qpos_des[k] = q_des.copy()
            qpos_act[k] = data.qpos.copy()
            data.ctrl[:] = q_des[7:]
            mujoco.mj_step(model, data)
            # data.qpos[:] = q_des
            # mujoco.mj_forward(model, data)  
            # mj_step computes forward dynamics (including kinematics) then
            # integrates qpos/qvel. After integration, derived quantities
            # (site_xpos, subtree_com, Jacobians) are stale — they reflect the
            # pre-integration state. Refresh them for the next IK solve.
            # configuration.update()

            elapsed = time.time() - start_time

            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


            backpack_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "backpack")
            current_com = data.subtree_com[backpack_body_id].copy()
            # Store position of CoM, left and right feet
            com_ref_pos[k] = com_target
            # com_pin_pos[k] = com_pin
            com_pb_pos[k] = current_com

            lf_ref_pos[k] = left_foot_path[k]
            # lf_pin_pos[k] = talos.data.oMf[talos.left_foot_id].translation
            lf_pb_pos[k]= data.site_xpos[left_foot_site_id]

            rf_ref_pos[k] = right_foot_path[k]
            # rf_pin_pos[k] = talos.data.oMf[talos.right_foot_id].translation
            rf_pb_pos[k] = data.site_xpos[right_foot_site_id]

            viewer.sync()

        plt.figure(figsize=(12, 8))
        plt.subplot(3, 3, 1)
        plt.plot(com_ref_pos[:, 0],label="CoM Reference-x", linestyle="--")
        plt.plot(com_pb_pos[:, 0] ,label="CoM Actual-x (MuJoCo)", linestyle="-")


        plt.legend()
        
        plt.subplot(3, 3, 2)
        plt.plot(com_ref_pos[:, 1],label="CoM Reference-y", linestyle="--")
        plt.plot(com_pb_pos[:, 1] ,label="CoM Actual-y (MuJoCo)", linestyle="-") 
        plt.legend()

        plt.subplot(3, 3, 3)
        plt.plot(com_ref_pos[:, 2],label="CoM Reference-z", linestyle="--")
        plt.plot(com_pb_pos[:, 2] ,label="CoM Actual-z (MuJoCo)", linestyle="-")

        plt.legend()

        plt.subplot(3, 3, 4)
        plt.plot(lf_ref_pos[:, 2], label="Left Foot Reference (z)", linestyle="--")
        plt.plot(lf_pb_pos[:, 2], label="Left Foot Actual (MuJoCo)", linestyle="-")
        plt.legend()
        plt.subplot(3, 3, 5)
        plt.plot(qpos_des[:, 8]- qpos_act[:, 8], label="L FE error", linestyle="--")
        plt.legend()
        plt.subplot(3, 3, 6)
        plt.plot(qpos_des[:, 9]- qpos_act[:, 9], label="L KNEE error", linestyle="--")
        plt.legend()
        plt.subplot(3, 3, 7)
        plt.plot(qpos_des[:, 7] - qpos_act[:, 7], label="L AA error", linestyle="--")
        plt.legend()
        plt.subplot(3, 3, 8)
        plt.plot(qpos_des[:, 10] - qpos_act[:, 10], label="LADPF error", linestyle="--")
        plt.legend()
        plt.subplot(3, 3, 9)
        plt.plot(qpos_des[:, 11] - qpos_act[:, 11], label="L IE error", linestyle="--")
        plt.legend()
        plt.show()
if __name__ == "__main__":
    main()