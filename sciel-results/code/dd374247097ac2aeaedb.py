import numpy as np

def construct():
    N = 50
    
    # Spherical initialization using Fibonacci lattice
    # This provides a more uniform distribution on a sphere than a grid subset
    # We choose a radius R_init such that there is room for a sphere of radius r ~ 0.13
    # Max r = 0.5 - R_init if aligned with axis. 
    # Let's pick R_init = 0.35. Max r = 0.15.
    R_init = 0.35
    
    # Fibonacci sphere
    points = np.zeros((N, 3))
    golden_angle = np.pi * (3. - np.sqrt(5.))
    for i in range(N):
        y = 1 - (i / (N - 1)) * 2
        radius = np.sqrt(1 - y*y)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points[i] = [x * R_init, y * R_init, z * R_init]
    
    # Optimization
    max_iter = 10000
    step_size = 0.005
    delta_active = 0.005
    
    # Use a fixed seed for any random operations if we add them, but here we are deterministic
    # centers = points.copy()
    centers = points.copy()
    
    for i in range(max_iter):
        # 1. Compute distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Wall distances
        abs_centers = np.abs(centers)
        max_coord = np.max(abs_centers, axis=1)
        wall_dists = 0.5 - max_coord
        
        # Current radius
        min_pair_dist = np.min(dist[~np.eye(N, dtype=bool)])
        min_wall_dist = np.min(wall_dists)
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        # Identify active constraints
        # Pairs with dist < 2 * (current_r + delta)
        active_pairs = np.argwhere(dist < 2 * (current_r + delta_active))
        active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
        
        # Walls with dist < current_r + delta_active
        active_walls = np.where(wall_dists < current_r + delta_active)[0]
        
        # Compute gradient
        grad = np.zeros_like(centers)
        
        # Pair contributions
        for i_idx, j_idx in active_pairs:
            v = centers[i_idx] - centers[j_idx]
            norm_v = np.linalg.norm(v)
            if norm_v > 1e-9:
                v_norm = v / norm_v
                grad[i_idx] += v_norm
                grad[j_idx] -= v_norm
        
        # Wall contributions
        for i_idx in active_walls:
            # Move away from the closest wall (towards center in the limiting coordinate)
            limiting_coords = np.argmax(abs_centers[i_idx])
            direction = -np.sign(centers[i_idx, limiting_coords])
            # Add a small push in the other coordinates too to help rotate/adjust?
            # Just the limiting coordinate is sufficient for the sub-gradient
            grad[i_idx, limiting_coords] += direction
            
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad /= grad_norm
            
        # Update
        centers += step_size * grad
        
        # Clamp to cube
        centers = np.clip(centers, -0.5, 0.5)
        
        # Decay step size
        step_size *= 0.9995
        delta_active *= 0.9995
        
    return centers
