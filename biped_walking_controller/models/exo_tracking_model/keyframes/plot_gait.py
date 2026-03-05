import csv
from pathlib import Path
import matplotlib.pyplot as plt
import jax.numpy as jnp

repo_dir = Path(__file__).resolve().parent
csv_path = "/home/jeenh/march-xi/ros2_ws/src/march_gpc/march_gpc/hydrax/hydrax/models/exo_tracking_model/keyframes/balanced_gait.csv"

csv_order = ["left_ankle_dpf", "left_ankle_ie", "left_hip_aa", "left_hip_fe", "left_knee",
                "right_ankle_dpf", "right_ankle_ie", "right_hip_aa", "right_hip_fe", "right_knee"]
qpos_order = ["left_hip_aa", "left_hip_fe", "left_knee", "left_ankle_dpf", "left_ankle_ie",
                "right_hip_aa", "right_hip_fe", "right_knee", "right_ankle_dpf", "right_ankle_ie"]

# Map CSV indices to qpos indices
reorder_indices = [csv_order.index(joint) for joint in qpos_order]
# Load reference trajectory directly from CSV
with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    data = [list(map(float, row)) for row in reader]

# Downsample by factor of 10
data = data[::10]

# Reorder columns to match qpos order
data = [[row[i] for i in reorder_indices] for row in data]

# Flip sign for hip_aa, ankle_dpf, and ankle_ie (indices 0, 3, 4, 5, 8, 9 in reordered data)
# Reordered order: left_hip_aa(0), left_hip_fe(1), left_knee(2), left_ankle_dpf(3), left_ankle_ie(4),
#                  right_hip_aa(5), right_hip_fe(6), right_knee(7), right_ankle_dpf(8), right_ankle_ie(9)
flip_indices = [ 3, 4, 8, 9]  # ankle_dpf, ankle_ie for both sides
for row in data:
    for idx in flip_indices:
        row[idx] *= -1

    fig, axes = plt.subplots(5, 2, figsize=(15, 10))
    plt.subplots_adjust(bottom=0.1)

    lines = []
    for i in range(len(qpos_order)):
        ax = axes[i // 2, i % 2]
        line, = ax.plot([row[i] for row in data], color='blue', alpha=0.5)
        ax.set_title(f'Joint name {qpos_order[i]}')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Qpos (radians)')
        lines.append(line)
    plt.show()