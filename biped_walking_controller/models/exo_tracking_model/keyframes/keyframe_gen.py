import csv         # number of keyframes
start = 0.0           # starting joint angle
step = 0.1
fps = 100       # step between keyframes
filename = "/home/jeenh/march-xi/ros2_ws/src/march_gpc/march_gpc/hydrax/hydrax/models/exo_tracking_model/keyframes/balanced_gait.xml"
csv_file = "/home/jeenh/march-xi/ros2_ws/src/march_gpc/march_gpc/hydrax/hydrax/models/exo_tracking_model/keyframes/gait_20260210_141747_interpolated_20260210_141747 (1).csv"

with open (csv_file, 'r') as csv_f:
    data = list(csv.reader(csv_f))
with open(filename, "w") as f:
    f.write("<mujoco>\n")
    f.write("  <keyframe>\n")
    for i in range(len(data)):
        if i == 0:
            continue
        elif i % 10 == 0:
            f.write(f"    <key name='spin_{i//10}' qpos='{' '.join(data[i//10])}' time='{i/fps}'/>\n")
    f.write("  </keyframe>\n</mujoco>\n")

print(f"Saved {filename} with {len(data)-1} keyframes.")
