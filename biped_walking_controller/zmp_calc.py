import numpy as np
def calculate_zmp(data, zmp_indices):
    """
    Calculate the Zero Moment Point (ZMP) position based on IMU sensor data.
    
    ZMP formulas:
    y_zmp = (Σ(m*(z_acc + g)*y_com) - Σ(m*y_acc*z_com) - Σ(I_x*omega_x)) / Σ(m*(z_acc + g))
    x_zmp = (Σ(m*(z_acc + g)*x_com) - Σ(m*x_acc*z_com) - Σ(I_y*omega_y)) / Σ(m*(z_acc + g))
    
    Args:
        data: MuJoCo data with current state
        zmp_indices: dict from precompute_zmp_indices() containing body_ids, acc_adrs, gyro_adrs
    
    Returns:
        np.array: [x_zmp, y_zmp, 0.001] - ZMP position on the floor
    """
    body_ids = zmp_indices['body_ids']
    acc_adrs = zmp_indices['acc_adrs']
    gyro_adrs = zmp_indices['gyro_adrs']
    
    """ Variables for zmp calculation """
    g = 9.81  # gravity
    m = np.array([3.205, 3.363, 3.363, 4.473, 4.473, 4.973, 4.973, 0.515, 0.515, 4.440, 4.440])  # mass for each body
    # took XX and YY of MOI of the sheet and switched them around to match mujoco standard
    I_x = np.array([0.1314, 0.1123, 0.1123, 0.4912, 0.4657, 0.2154, 0.2153, 0.0024, 0.0024, 0.0470, 0.0470])  # inertia around x-axis for each body
    I_y = np.array([0.0440, 0.0836, 0.0736, 0.4654, 0.4436, 0.2200, 0.2154, 0.0026, 0.0027, 0.1909, 0.1928])  # inertia around y-axis for each body
    
    # Get COM positions for all bodies (world frame)
    # xipos shape is (nbody, 3)
    x_com = data.xipos[body_ids, 0]
    y_com = data.xipos[body_ids, 1]
    z_com = data.xipos[body_ids, 2]
    
    # Get rotation matrices for all bodies
    # xmat shape is (nbody, 9) - need to reshape to (nbody, 3, 3)
    body_xmats = data.xmat[body_ids].reshape(-1, 3, 3)
    
    # Get accelerometer and gyro data for all bodies
    # Each sensor has 3 consecutive values in sensordata
    acc_sensors = np.array([data.sensordata[adr:adr+3] for adr in acc_adrs])  # shape: (n_bodies, 3)
    gyro_sensors = np.array([data.sensordata[adr:adr+3] for adr in gyro_adrs])  # shape: (n_bodies, 3)
    
    # Transform accelerations to world frame: acc_world = body_xmat @ acc_sensor
    # Using einsum for batched matrix-vector multiplication
    acc_world = np.einsum('bij,bj->bi', body_xmats, acc_sensors)  # shape: (n_bodies, 3)
    x_acc = acc_world[:, 0]
    y_acc = acc_world[:, 1]
    z_acc = acc_world[:, 2]
    
    # Transform angular velocity to world frame
    omega_world = np.einsum('bij,bj->bi', body_xmats, gyro_sensors)  # shape: (n_bodies, 3)
    alpha_x = omega_world[:, 0]  # Using angular velocity instead of acceleration
    alpha_y = omega_world[:, 1]
    
    # Denominator (same for both x and y)
    denominator = np.sum(m * (z_acc + g))

    # # Debug logging
    # print(f"ZMP denom={denominator}")
    # print(f"ZMP z_acc+g min/max={np.min(z_acc + g)}/{np.max(z_acc + g)}")
    # print(f"ZMP x_acc/y_acc/z_acc means={np.mean(x_acc)}/{np.mean(y_acc)}/{np.mean(z_acc)}")
    
    # Calculate y_zmp numerator
    y_zmp_numerator = (np.sum(m * (z_acc + g) * y_com) - 
                       np.sum(m * y_acc * z_com) - 
                       np.sum(I_x * alpha_x))
    
    # Calculate x_zmp numerator
    x_zmp_numerator = (np.sum(m * (z_acc + g) * x_com) - 
                       np.sum(m * x_acc * z_com) - 
                       np.sum(I_y * alpha_y))
    
    # Avoid division by zero using np.where
    safe_denominator = np.where(np.abs(denominator) < 1e-6, 1.0, denominator)
    is_valid = np.abs(denominator) >= 1e-6
    # print(f"is_valid={is_valid}")
    y_zmp = np.where(is_valid, y_zmp_numerator / safe_denominator, 0.0)
    x_zmp = np.where(is_valid, x_zmp_numerator / safe_denominator, 0.0)
    # print(f"ZMP x_zmp={x_zmp}, y_zmp={y_zmp}")
    
    return np.array([x_zmp, y_zmp, 0.001])  # Slightly above floor for visibility