import time
import mujoco
import mujoco.viewer
import csv
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from mujoco import mjx

mj_model = mujoco.MjModel.from_xml_path('/home/jeenh/march-xi/ros2_ws/src/march_gpc/march_gpc/hydrax/hydrax/models/exo_tracking_model/exo_tracking_ground.xml')
reference = mj_model.key_qpos
reference[:,2] += 0.03 # add small height offset to avoid initial penetration
print("First reference value: ", reference[0])
print("Reference shape: ", reference.shape)
reference_fps = 10 # Hz
# Get sensor IDs
left_foot_pos_sensor = mujoco.mj_name2id(
    mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "left_foot_position"
)
left_foot_quat_sensor = mujoco.mj_name2id(
    mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "left_foot_orientation"
)
right_foot_pos_sensor = mujoco.mj_name2id(
    mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "right_foot_position"
)
right_foot_quat_sensor = mujoco.mj_name2id(
    mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "right_foot_orientation"
)
# Precompute reference foot positions and orientations
mj_data = mujoco.MjData(mj_model)
n_frames = len(reference)
ref_left_pos = np.zeros((n_frames, 3))
ref_left_quat = np.zeros((n_frames, 4))
ref_right_pos = np.zeros((n_frames, 3))
ref_right_quat = np.zeros((n_frames, 4))

keyframe_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
mj_data.qpos[:] = mj_model.key_qpos[keyframe_id]
com_sensor = "torso_subtreecom"
com_sensor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, com_sensor)
com_sensor_adr = mj_model.sensor_adr[com_sensor_id]

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        viewer.sync()
        # mj_data.qpos[:] = reference[0]
        # mujoco.mj_forward(mj_model, mj_data)
        # for i in range(n_frames):
        #     mj_data.qpos[:] = reference[i]
        #     viewer.sync()
        #     mujoco.mj_step(mj_model, mj_data)
        #     ref_left_pos[i] = mj_data.site_xpos[mj_model.site("pos_L_foot").id]
        #     ref_right_pos[i] = mj_data.site_xpos[mj_model.site("pos_R_foot").id]
        #     mujoco.mju_mat2Quat(
        #         ref_left_quat[i],
        #         mj_data.site_xmat[mj_model.site("pos_L_foot").id].flatten(),
        #     )  
        #     mujoco.mju_mat2Quat(
        #         ref_right_quat[i],
        #         mj_data.site_xmat[mj_model.site("pos_R_foot").id].flatten(),
        #     )   


