"""
Convert a qpos CSV (with one column layout) to a MuJoCo keyframe XML file.

The CSV has columns: elapsed time, timestamp, topic, value
where topic is like /qpos.qpos[0], /qpos.qpos[1], etc.
All samples for qpos[0] come first, then qpos[1], etc.

Usage:
    python csv_to_keyframe_xml.py <input_csv> <output_xml> [--dt 0.1] [--name-prefix spin]
"""

import argparse
import csv
import re
import numpy as np
from collections import defaultdict


def parse_csv(csv_path):
    """Parse the CSV and return {qpos_index: [(elapsed_time, value), ...]}."""
    data = defaultdict(list)

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header

        for row in reader:
            elapsed_time = float(row[0])
            topic = row[2]
            value = float(row[3])

            # Extract qpos index from topic like /qpos.qpos[3]
            match = re.search(r"qpos\[(\d+)\]", topic)
            if match:
                idx = int(match.group(1))
                data[idx].append((elapsed_time, value))

    return data


def resample_uniform(data, dt):
    """
    Resample all qpos signals onto a uniform time grid with spacing dt.
    Returns (times, qpos_array) where qpos_array is shape (n_steps, n_joints).
    """
    n_joints = max(data.keys()) + 1

    # Find common time range
    t_min = max(samples[0][0] for samples in data.values())
    t_max = min(samples[-1][0] for samples in data.values())

    times = np.arange(t_min, t_max, dt)
    # Round to avoid floating point drift
    times = np.round(times, 6)

    qpos_array = np.zeros((len(times), n_joints))

    for idx in range(n_joints):
        raw_times = np.array([t for t, v in data[idx]])
        raw_values = np.array([v for t, v in data[idx]])
        qpos_array[:, idx] = np.interp(times, raw_times, raw_values)

    return times, qpos_array


def write_keyframe_xml(output_path, times, qpos_array, name_prefix, dt):
    """Write the keyframe XML file."""
    with open(output_path, "w") as f:
        f.write("<mujoco>\n")
        f.write("  <keyframe>\n")

        for i, (t, qpos) in enumerate(zip(times, qpos_array)):
            name = f"{name_prefix}_{i + 1}"
            qpos_str = " ".join(f"{v}" for v in qpos)
            time_val = round((i + 1) * dt, 6)
            f.write(
                f"    <key name='{name}' qpos='{qpos_str}' time='{time_val}'/>\n"
            )

        f.write("  </keyframe>\n")
        f.write("</mujoco>\n")

    print(f"Wrote {len(times)} keyframes to {output_path}")
    print(f"  Joints: {qpos_array.shape[1]}")
    print(f"  Time range: {times[0]:.3f} - {times[-1]:.3f}s (dt={dt}s)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert qpos CSV to MuJoCo keyframe XML"
    )
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("output_xml", help="Path to output XML file")
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Time step between keyframes in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--name-prefix",
        default="spin",
        help="Prefix for keyframe names (default: spin)",
    )
    args = parser.parse_args()

    print(f"Reading {args.input_csv}...")
    data = parse_csv(args.input_csv)
    print(f"  Found {len(data)} qpos indices, {len(next(iter(data.values())))} samples each")

    print(f"Resampling at dt={args.dt}s...")
    times, qpos_array = resample_uniform(data, args.dt)

    write_keyframe_xml(args.output_xml, times, qpos_array, args.name_prefix, args.dt)


if __name__ == "__main__":
    main()
