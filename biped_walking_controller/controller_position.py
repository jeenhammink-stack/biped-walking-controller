"""Author: MARK EN JEEN # LOL."""

import numpy as np
import mujoco
def low_level_update(model, data, joint_gains: dict, desired_pos: np.ndarray) -> None:


    actuator_names = [model.actuator(i).name for i in range(model.nu)]

    for idx, name in enumerate(actuator_names):
        if joint_gains is not None:

            kp_val = joint_gains[name.upper()]["kp"]
            kd_val = joint_gains[name.upper()]["kd"]
        else:
            kp_val = 50
            kd_val = 0.8

        joint_id = model.actuator_trnid[idx, 0]  # get joint of the actuator
        current_pos = data.qpos[model.jnt_qposadr[joint_id]]
        current_vel = data.qvel[model.jnt_dofadr[joint_id]]
        # Set cascaded PD control
        data.ctrl[idx] = np.clip(kp_val * (desired_pos[idx] - current_pos) - kd_val * current_vel, -1, 1)
