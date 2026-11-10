import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    max_time = 115.0

    # 1. Initialization: Hexagonal Lattice
    # Generate hex points
    # dx = 1, dy = sqrt(3)/2
    # We want points in [-0.5, 0.5]
    # Let's generate a grid of hex points
    # Scale: We want about 50 points.
    # Area = 1. Area per hex point ~ sqrt(3)/2 * (spacing)^2.
    # If we want density 50/1 = 50, spacing^2 ~ 1/(50 * sqrt(3)/2) ~ 0.023.
    # spacing ~ 0.15.
    
    spacing = 0.15
    dx = spacing
    dy = spacing * np.sqrt(3) / 2
    
    x_min, x_max = -0.5, 0.5
    y_min, y_max = -0.5, 0.5
    
    x_vals = np.arange(x_min, x_max + dx, dx)
    y_vals = np.arange(y_min, y_max + dy, dy)
    
    hex_points = []
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            # Offset every other column
            y_offset = (dy / 2) if i % 2 == 1 else 0
            hex_points.append((x, y + y_offset))
            
    hex_points = np.array(hex_points)
    
    # Filter points inside the square
    mask = (hex_points[:, 0] >= -0.5) & (hex_points[:, 0] <= 0.5) & \
           (hex_points[:, 1] >= -0.5) & (hex_points[:, 1] <= 0.5)
    hex_points = hex_points[mask]
    
    # If we have more than N, pick N closest to center? Or just take first N?
    # Taking first N from the generation order (row by row) is biased.
    # Let's pick N points that are "representative".
    # For simplicity, if len(hex_points) >= N, we can just take a random subset or the first N.
    # A better initialization is to take the N points with the highest "min distance to other points" in the hex grid?
    # That's expensive.
    # Let's just take a random subset of the hex points if there are more than N.
    if len(hex_points) > N:
        idx = np.random.choice(len(hex_points), N, replace=False)
        centers = hex_points[idx].copy()
    else:
        # Pad with random points if needed (unlikely)
        centers = hex_points.copy()
        while len(centers) < N:
            centers = np.vstack([centers, np.random.uniform(-0.5, 0.5, (1, 2))])
            
    # Add jitter
    centers += np.random.uniform(-0.01, 0.01, centers.shape)
    centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
    centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
    
    # 2. Optimization: Gradient Descent on Sum(1/d)
    eta = 0.005
    max_iters = 30000
    
    for i in range(max_iters):
        if time.time() - start_time > max_time:
            break
            
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion: Force on i from j is -(p_i - p_j) / d^3
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=-1)
        dists_sq[dists_sq < 1e-12] = 1e-12
        dists = np.sqrt(dists_sq)
        
        # factor[i, j] = -1 / d^3
        factor = -1.0 / (dists**3)
        np.fill_diagonal(factor, 0)
        
        forces += np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary forces
        x = centers[:, 0]
        y = centers[:, 1]
        
        fx_bound = np.zeros_like(x)
        fy_bound = np.zeros_like(y)
        
        # Soft boundary penalty: (max(0, x-0.5)^2 + ...)
        # Gradient: 2*(x-0.5) if x>0.5
        mask_r = x > 0.5
        fx_bound[mask_r] = 2.0 * (x[mask_r] - 0.5)
        mask_l = x < -0.5
        fx_bound[mask_l] = -2.0 * (x[mask_l] + 0.5)
        
        mask_t = y > 0.5
        fy_bound[mask_t] = 2.0 * (y[mask_t] - 0.5)
        mask_b = y < -0.5
        fy_bound[mask_b] = -2.0 * (y[mask_b] + 0.5)
        
        forces[:, 0] += fx_bound
        forces[:, 1] += fy_bound
        
        # Update
        centers -= eta * forces
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        # Decay
        eta *= 0.9995
        
    return centers
