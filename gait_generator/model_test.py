import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from biped_walking_controller.model_mujoco import Exo

model = Exo().mj_model_torque
data = mujoco.MjData(model)
data.qpos = model.keyframe("home").qpos
# data.xpos[model.body("L_foot").id] = np.array([2.0, 1.0, 5.0])
mujoco.mj_forward(model, data)
x_offset = data.xpos[model.body("L_foot").id][0]
print("Initial left foot position:", data.xpos[model.body("L_foot").id])
data.qpos[0] = data.qpos[0] - x_offset
print("Adjusted qpos[0]:", data.qpos[0])
mujoco.mj_forward(model, data) 

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_forward(model, data)
        viewer.sync()
        print("Left foot position:", data.xpos[model.body("L_foot").id])