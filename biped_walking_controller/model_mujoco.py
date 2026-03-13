from pathlib import Path

import numpy as np
# import pinocchio as pin
import mujoco


# # def print_joints(model):
# #     for j_id, j_name in enumerate(model.names):
# #         print(j_id, j_name, model.joints[j_id].shortname(), model.joints[j_id].nq)
# def print_joints(model):


# def print_frames(model):
#     for i, frame in enumerate(model.frames):
#         print(i, frame.name, frame.parent, frame.type)


# def set_joint(q, model, joint_name, val):
#     joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
#     if joint_id > 0 and model.joints[joint_id].nq == 1:
#         q[model.joints[joint_id].idx_q] = val


# def q_from_base_and_joints(q, oMb):
#     R = oMb.rotation
#     p = oMb.translation
#     quat = pin.Quaternion(R)  # xyzw
#     q_out = q.copy()
#     q_out[:3] = p
#     q_out[3:7] = np.array([quat.x, quat.y, quat.z, quat.w])
#     return q_out


class Exo:
    def __init__(self):
        # Load full model
        # path_to_urdf = path_to_model / "talos_data" / "urdf" / "talos_full.urdf"
        # full_model, full_col_model, full_vis_model = pin.buildModelsFromUrdf(
        #     path_to_urdf, str(path_to_model), pin.JointModelFreeFlyer()
        # )
        repo_dir = Path(__file__).resolve().parent
        xml_path__position = str(repo_dir / "models" / "exo_tracking_model" / "exo_tracking_position.xml")
        xml_path_torque = str(repo_dir / "models" / "exo_tracking_model" / "exo_tracking_torque.xml")
        xml_path_air = str(repo_dir / "models" / "exo_tracking_model" / "exo_tracking_air.xml")
        self.mj_model_position = mujoco.MjModel.from_xml_path(xml_path__position)
        self.mj_model_air = mujoco.MjModel.from_xml_path(xml_path_air)
        self.mj_model_torque = mujoco.MjModel.from_xml_path(xml_path_torque)
        # Position the arms
        # We lock joints of the upper body since there are not meant to move with LIPM model
        # set_joint(q, full_model, "leg_left_4_joint", 0.0)
        # set_joint(q, full_model, "leg_right_4_joint", 0.0)
        # set_joint(q, full_model, "arm_right_4_joint", -1.5)
        # set_joint(q, full_model, "arm_left_4_joint", -1.5)

        # We build a reduced model by locking the specificied joints if needed
        # self.reduced = reduced
        # if self.reduced:
        #     joints_to_lock = list(self.get_locked_joints_idx())
        #     self.model, self.geom = pin.buildReducedModel(
        #         full_model, full_col_model, joints_to_lock, q
        #     )
        #     _, self.vis = pin.buildReducedModel(full_model, full_vis_model, joints_to_lock, q)
        # else:
        #     self.model = full_model
        #     self.geom = full_col_model
        #     self.vis = full_vis_model

        # self.mj_data = mujoco.MjData(self.mj_model)

        # upper_v_idx = {}
        # for j in self.model.joints:
        #     if j.nv == 0 or j.idx_q == -1:
        #         continue

        #     name = self.model.names[j.id]  # joint name from URDF
        #     # if upper_rx.match(name):
        #     start = j.idx_v  # first velocity index for this joint
        #     upper_v_idx[name] = (start, j.idx_q)

        # self.left_foot_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "pos_L_foot")
        # self.right_foot_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "pos_R_foot")
        # self.backpack_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "imu_backpack")

    def set_and_get_default_pose(self):
        # Initialize reduced model
        q_pos = self.mj_model.keyframe("home").qpos
        self.mj_data.qpos[:] = q_pos

        # if not self.reduced:
        #     set_joint(q, self.model, "leg_left_4_joint", 0.0)
        #     set_joint(q, self.model, "leg_right_4_joint", 0.0)
        #     set_joint(q, self.model, "arm_right_4_joint", -1.5)
        #     set_joint(q, self.model, "arm_left_4_joint", -1.5)

        # # Initialize left leg position
        # set_joint(q, self.model, "leg_left_1_joint", 0.0)
        # set_joint(q, self.model, "leg_left_2_joint", 0.0)
        # set_joint(q, self.model, "leg_left_3_joint", -0.5)
        # set_joint(q, self.model, "leg_left_4_joint", 1.0)
        # set_joint(q, self.model, "leg_left_5_joint", -0.6)

        # # Initialize right leg position
        # set_joint(q, self.model, "leg_right_1_joint", 0.0)
        # set_joint(q, self.model, "leg_right_2_joint", 0.0)
        # set_joint(q, self.model, "leg_right_3_joint", -0.5)
        # set_joint(q, self.model, "leg_right_4_joint", 1.0)
        # set_joint(q, self.model, "leg_right_5_joint", -0.6)

        # Update position of the model and the data
        # pin.forwardKinematics(self.model, self.data)
        # pin.updateFramePlacements(self.model, self.data)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return q_pos

    # def get_joint_id(self, name):
    #     jid = self.model.getJointId(name)

    #     n_joints = len(self.model.joints)

    #     return self.model.joints[jid].idx_q if jid < n_joints else None
    def get_joint_id(self, name):
        jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return self.mj_model.jnt_qposadr[jid] if jid >= 0 else None
    
    @staticmethod
    def get_locked_joints_idx():
        return range(14, 46)

    @staticmethod
    def get_rf_link_name():
        return "leg_right_6_link"

    @staticmethod
    def get_lf_link_name():
        return "leg_left_6_link"
