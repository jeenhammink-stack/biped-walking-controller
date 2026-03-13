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
from matplotlib import pyplot as plt
from sympy import fps

from biped_walking_controller.foot import (
    compute_feet_trajectories,
    BezierCurveFootPathGenerator,
    compute_steps_sequence,
)

from biped_walking_controller.inverse_kinematic_mujoco import solve_inv_kinematics_mujoco, InvKinSolverParamsMujoco
from biped_walking_controller.model_mujoco import Exo

from biped_walking_controller.preview_control import (
    PreviewControllerParams,
    compute_preview_control_matrices,
    update_control,
    compute_zmp_ref,
    cubic_spline_interpolation,
)
from biped_walking_controller.controller_position import low_level_update



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
    general_params.t_ds = 0.2

    general_params.n_steps = 5
    general_params.l_stride = 0.15
    general_params.max_height_foot = 0.03 # Lower foot height
    general_params.n_solver_iter = 1500

    ctrler_params.n_preview_steps = int(round(general_params.t_preview / general_params.dt))
    return general_params, ctrler_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--make-keyframes", nargs="?", const=True, default=False, metavar="PATH", help="Generate keyframes for the MuJoCo model. Optionally specify an output file path.")
    args = p.parse_args()

    scen_params, ctrler_params = get_accurate_sim_params()
    
    # model = Exo().mj_model_torque
    model = Exo().mj_model_torque
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
        damping_task := mink.DampingTask(model=model, cost=2.0)
        ]
    ik_sol_params = InvKinSolverParamsMujoco(
        fixed_foot=left_foot_name,
        swing_foot=right_foot_name,
        dt=model.opt.timestep,
    )

    ##############
    # LIPM setup #
    ##############
    right_foot_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pos_R_foot")
    left_foot_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pos_L_foot")
    right_foot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_foot")
    left_foot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_foot")

    backpack_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu_backpack")
    # Step sim couple of times to get accurate initial positions


    right_foot_pos_target = data.site_xpos[right_foot_site_id].copy()
    print(f"Initial right foot position from MuJoCo: {right_foot_pos_target}")
    right_foot_quat_target = data.xquat[right_foot_body_id].copy()

    left_foot_pos_target = data.site_xpos[left_foot_site_id].copy()
    print(f"Initial left foot position from MuJoCo: {left_foot_pos_target}")
    left_foot_quat_target = data.xquat[left_foot_body_id].copy()
    mujoco.mj_forward(model, data)  # Update all dependent quantities after changing qpos
    print(f"Adjusted initial right foot position from MuJoCo: {data.xpos[right_foot_body_id]}, Adjusted initial left foot position from MuJoCo: {data.xpos[left_foot_body_id]}")
    backpack_site_pos = data.site_xpos[backpack_site_id].copy()
    backpack_angle = 3.0  # degrees

    feet_mid = 0.5 * (right_foot_pos_target + left_foot_pos_target)
    des_height = 0.61 # m
    com_initial_target = np.array([feet_mid[0]+0.05, feet_mid[1], des_height])  # Initial CoM target above the midpoint of the feet.
    backpack_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "backpack")
    # Get the subtree COM for the backpack body (includes all child bodies)
    com_initial = data.subtree_com[backpack_body_id].copy()
    print(f"Initial CoM position from MuJoCo: {com_initial}, Initial backpack position: {backpack_site_pos}")
    ctrler_mat = compute_preview_control_matrices(ctrler_params, scen_params.dt)

    steps_pose, steps_ids = compute_steps_sequence(
        rf_initial_pose=right_foot_pos_target,
        lf_initial_pose=left_foot_pos_target,
        n_steps=scen_params.n_steps,
        l_stride=scen_params.l_stride,
    )


    t, left_foot_path, right_foot_path, phases = compute_feet_trajectories(
        rf_initial_pose=right_foot_pos_target,
        lf_initial_pose=left_foot_pos_target,
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

    x_k = np.array([0.0, com_initial_target[0], 0.0, 0.0], dtype=float)
    y_k = np.array([0.0, com_initial_target[1], 0.0, 0.0], dtype=float)

    keyframes = np.zeros((len(phases), model.nq))

    com_ref_pos = np.zeros((len(phases), 3))
    com_muj_pos = np.zeros((len(phases), 3))

    lf_ref_pos = np.zeros((len(phases), 3))
    lf_muj_pos = np.zeros((len(phases), 3))
    lf_ik_pos = np.zeros((len(phases), 3))

    rf_ref_pos = np.zeros((len(phases), 3))
    rf_muj_pos = np.zeros((len(phases), 3))
    


    qpos_des = np.zeros((len(phases), model.nq))
    qpos_act = np.zeros((len(phases), model.nq))

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

    data_copy = mujoco.MjData(model)
    data_copy.qpos[:] = model.keyframe("home").qpos.copy()
    mujoco.mj_forward(model, data_copy)  # Update all dependent quantities after changing qpos

    ##################
    # RUN SIMULATION #
    ##################

    """Note on the control loop structure:
    - The main loop iterates over the precomputed foot trajectories and phases.
    - At each iteration, it computes the desired CoM target based on the LIPM preview control.
    - It then solves the inverse kinematics to get the desired joint positions for the current time step.
    - Timing is important here. Controls are applied for as long the timestep lasts """

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer launched successfully.")
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        for k, _ in enumerate(phases[:-2]):
            start_time = time.time()
            zmp_ref_horizon = zmp_padded[k + 1 : k + ctrler_params.n_preview_steps]

            _, x_k, y_k = update_control(
                ctrler_mat, zmp_padded[k], zmp_ref_horizon, x_k.copy(), y_k.copy()
            )

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
                    stance_des_pos=right_foot_pos_target,
                    stance_des_quat=right_foot_quat_target,
                    swing_des_pos=left_foot_path[k],
                    swing_des_quat=left_foot_quat_target,
                    backpack_angle=backpack_angle,
                    params=ik_sol_params,
                )
                
                if phases[k + 1] > 0.0:
                # Update the stance target so that u dont use the value of one gait cycle before
                    left_foot_pos_target = left_foot_path[k + 1].copy()
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
                    stance_des_pos=left_foot_pos_target,
                    stance_des_quat=left_foot_quat_target,
                    swing_des_pos=right_foot_path[k],
                    swing_des_quat=right_foot_quat_target,
                    backpack_angle=backpack_angle,
                    params=ik_sol_params,
                )

                if phases[k + 1] < 0.0:
                    right_foot_pos_target = right_foot_path[k + 1].copy()
            # data_copy.qpos[:] = q_des
            # mujoco.mj_forward(model, data_copy)  # Update all dependent quantities after changing qpos
            lf_ik_pos[k] = data_copy.site_xpos[left_foot_site_id]
            keyframes[k] = q_des.copy()
            qpos_des[k] = q_des.copy()
            qpos_act[k] = data.qpos.copy()
            low_level_update(model, data, joint_gains=joint_gains, desired_pos=q_des[7:])
            # data.ctrl[:] = q_des[7:].copy()  # Open loop control to track the IK solution

            mujoco.mj_step(model, data)
            # data.qpos[:] = q_des
            # mujoco.mj_forward(model, data)  
            # mj_step computes forward dynamics (including kinematics) then
            # integrates qpos/qvel. After integration, derived quantities
            # (site_xpos, subtree_com, Jacobians) are stale — they reflect the
            # pre-integration state. Refresh them for the next IK solve.
            # configuration.update()

            elapsed = time.time() - start_time

            if elapsed < scen_params.dt:
                time.sleep(scen_params.dt - elapsed)


            backpack_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "backpack")
            current_com = data.subtree_com[backpack_body_id].copy()
            # Store position of CoM, left and right feet
            com_ref_pos[k] = com_target
            # com_pin_pos[k] = com_pin
            com_muj_pos[k] = current_com

            lf_ref_pos[k] = left_foot_path[k]
            # lf_pin_pos[k] = talos.data.oMf[talos.left_foot_id].translation
            lf_muj_pos[k]= data.site_xpos[left_foot_site_id]

            rf_ref_pos[k] = right_foot_path[k]
            # rf_pin_pos[k] = talos.data.oMf[talos.right_foot_id].translation
            rf_muj_pos[k] = data.site_xpos[right_foot_site_id]

            viewer.sync()

        if args.make_keyframes:
            if isinstance(args.make_keyframes, str):
                filename = Path(args.make_keyframes)
            else:
                filename = Path(__file__).parent / "stable_gait.xml"
            with open (filename, "w") as f:
                f.write("<mujoco>\n")
                f.write("  <keyframe>\n")
                for i in range(keyframes.shape[0]):
                    f.write(f"    <key name='spin_{i}' qpos='{float(keyframes[i, 0])} {float(keyframes[i, 1])} {float(keyframes[i, 2])} {float(keyframes[i, 3])} {float(keyframes[i, 4])} {float(keyframes[i, 5])} {float(keyframes[i, 6])} {float(keyframes[i, 7])} {float(keyframes[i, 8])} {float(keyframes[i, 9])} {float(keyframes[i, 10])} {float(keyframes[i, 11])} {float(keyframes[i, 12])} {float(keyframes[i, 13])} {float(keyframes[i, 14])} {float(keyframes[i, 15])} {float(keyframes[i, 16])}' />\n") 
                f.write("  </keyframe>\n</mujoco>\n")
            print(f"Keyframes written to {filename}")

        plotlenght = 1540
        plt.figure(figsize=(40, 30))
        plt.subplot(4, 4, 1)
        plt.plot(com_ref_pos[:plotlenght, 0],label="CoM Reference-x", linestyle="--")
        plt.plot(com_muj_pos[:plotlenght, 0] ,label="CoM Actual-x (MuJoCo)", linestyle="-")


        plt.legend()
        
        plt.subplot(4, 4, 2)
        plt.plot(com_ref_pos[:plotlenght, 1],label="CoM Reference-y", linestyle="--")
        plt.plot(com_muj_pos[:plotlenght, 1] ,label="CoM Actual-y (MuJoCo)", linestyle="-") 
        plt.legend()

        plt.subplot(4, 4, 3)
        plt.plot(com_ref_pos[:plotlenght, 2],label="CoM Reference-z", linestyle="--")
        plt.plot(com_muj_pos[:plotlenght, 2] ,label="CoM Actual-z (MuJoCo)", linestyle="-")

        plt.legend()

        plt.subplot(4, 4, 4)
        plt.plot(lf_ref_pos[:plotlenght, 2], label="Left Foot Reference (z)", linestyle="--")
        plt.plot(lf_muj_pos[:plotlenght, 2], label="Left Foot Actual (MuJoCo)", linestyle="-")
        plt.plot(lf_ik_pos[:plotlenght, 2], label="Left Foot IK Solution (z)", linestyle=":")
        plt.legend()
        plt.subplot(4, 4, 5)
        plt.plot(qpos_des[:plotlenght, 8], label="L FE desired", linestyle="--")
        plt.plot(qpos_act[:plotlenght, 8], label="L FE actual", linestyle="-")
        plt.legend()
        plt.subplot(4, 4, 6)
        plt.plot(qpos_des[:plotlenght, 9], label="L KNEE desired", linestyle="--")
        plt.plot(qpos_act[:plotlenght, 9], label="L KNEE actual", linestyle="-")
        plt.legend()
        plt.subplot(4, 4, 7)
        plt.plot(qpos_des[:plotlenght, 7], label="L AA desired", linestyle="--")
        plt.plot(qpos_act[:plotlenght, 7], label="L AA actual", linestyle="-")
        plt.legend()
        plt.subplot(4, 4, 8)
        plt.plot(qpos_des[:plotlenght, 10], label="L ADPF desired", linestyle="--")
        plt.plot(qpos_act[:plotlenght, 10], label="L ADPF actual", linestyle="-")
        plt.legend()
        plt.subplot(4, 4, 9)
        plt.plot(qpos_des[:plotlenght, 11], label="L IE desired", linestyle="--")
        plt.plot(qpos_act[:plotlenght, 11], label="L IE actual", linestyle="-")
        plt.legend()
        plt.subplot(4, 4, 10)
        plt.plot(qpos_des[:plotlenght, 13], label="R FE desired", linestyle="--")
        plt.plot(qpos_act[:plotlenght, 13], label="R FE actual", linestyle="-")
        plt.legend()
        plt.subplot(4, 4, 11)
        plt.plot(qpos_des[:plotlenght, 14], label="R KNEE desired", linestyle="--")
        plt.plot(qpos_act[:plotlenght, 14], label="R KNEE actual", linestyle="-")
        plt.legend()
        plt.subplot(4, 4, 12)
        plt.plot(qpos_des[:plotlenght, 12], label="R AA desired", linestyle="--")
        plt.plot(qpos_act[:plotlenght, 12], label="R AA actual", linestyle="-")
        plt.legend()
        plt.subplot(4, 4, 13)
        plt.plot(qpos_des[:plotlenght, 15], label="R ADPF desired", linestyle="--")
        plt.plot(qpos_act[:plotlenght, 15], label="R ADPF actual", linestyle="-")
        plt.legend()
        plt.subplot(4, 4, 14)
        plt.plot(qpos_des[:plotlenght, 16], label="R IE desired", linestyle="--")
        plt.plot(qpos_act[:plotlenght, 16], label="R IE actual", linestyle="-")
        plt.legend()
        plt.show()
if __name__ == "__main__":
    main()