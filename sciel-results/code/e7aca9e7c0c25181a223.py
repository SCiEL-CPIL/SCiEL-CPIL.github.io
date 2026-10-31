import numpy as np
import time
import random

N = 21
L = 1.0  # Side length of the square, perimeter = 4*L = 4

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Initial Setup
    # Start with a 5x5 grid arrangement, picking 21 cells.
    # This ensures a decent starting point where circles are well-separated.
    side = 5
    cell_size = L / side
    # Indices for 21 cells in a 5x5 grid (row-major)
    indices = []
    for i in range(side):
        for j in range(side):
            indices.append((i, j))
            if len(indices) == N:
                break
        if len(indices) == N:
            break
    
    # Initial positions (centers) and radii
    # Radius is slightly less than half cell size to allow movement
    init_r = cell_size * 0.49
    centers = np.zeros((N, 2))
    radii = np.zeros(N)
    
    for k, (i, j) in enumerate(indices):
        centers[k, 0] = (j + 0.5) * cell_size
        centers[k, 1] = (i + 0.5) * cell_size
        radii[k] = init_r
        
    # Helper function to calculate validity and objective
    def evaluate(pos, rad):
        # Check bounds
        # x in [r, L-r], y in [r, L-r]
        # If out of bounds, we need to shrink or move.
        # For the objective, we want to maximize sum(rad) - penalty.
        
        # Boundary violations
        # min_dist_to_wall = min(x, y, L-x, L-y)
        # If rad > min_dist_to_wall, violation = rad - min_dist_to_wall
        dist_x = np.minimum(pos[:, 0], L - pos[:, 0])
        dist_y = np.minimum(pos[:, 1], L - pos[:, 1])
        min_dist_wall = np.minimum(dist_x, dist_y)
        
        bound_viol = np.maximum(0, rad - min_dist_wall)
        
        # Pairwise violations
        # d_ij = distance between centers
        # violation = max(0, r_i + r_j - d_ij)
        # Vectorized distance calculation
        # diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # dist = np.linalg.norm(diff, axis=-1)
        
        # To save memory/time for small N, we can do it in a loop or vectorized.
        # N=21 is small. Vectorized is faster in Python.
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        
        # Sum of radii pairs
        sum_rad = rad[:, np.newaxis] + rad[np.newaxis, :]
        
        viol_mat = np.maximum(0, sum_rad - dist)
        # Sum only upper triangle to avoid double counting
        # Or just sum all and divide by 2? No, penalty should be symmetric.
        # Let's sum the upper triangle.
        iu = np.triu_indices(N, k=1)
        pair_viol = np.sum(viol_mat[iu])
        
        total_viol = np.sum(bound_viol) + pair_viol
        obj = np.sum(rad) - 100.0 * total_viol # Heavy penalty for violations
        
        return obj, total_viol

    current_pos, current_rad = centers.copy(), radii.copy()
    current_obj, current_viol = evaluate(current_pos, current_rad)
    
    best_pos = current_pos.copy()
    best_rad = current_rad.copy()
    best_obj = current_obj
    best_viol = current_viol
    
    # Simulated Annealing Parameters
    T = 0.05
    T_min = 1e-6
    alpha = 0.9995
    max_iter = 200000 # We will check time in loop
    
    # Pre-compute indices for pairwise
    iu = np.triu_indices(N, k=1)
    i_idx, j_idx = iu
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # 1. Move a random center
        k = random.randint(0, N-1)
        # Gaussian step
        step = 0.02 * T * 100 # Scale step with temperature
        new_pos = current_pos.copy()
        new_pos[k, 0] += random.gauss(0, step)
        new_pos[k, 1] += random.gauss(0, step)
        
        # 2. Optionally adjust a random radius
        if random.random() < 0.5:
            new_rad = current_rad.copy()
            # Try to increase radius slightly, or decrease if needed
            delta_r = random.gauss(0, 0.005)
            new_rad[k] += delta_r
            # Clamp to positive
            if new_rad[k] < 0.01:
                new_rad[k] = 0.01
        else:
            new_rad = current_rad
        
        # Evaluate
        new_obj, new_viol = evaluate(new_pos, new_rad)
        
        # Acceptance
        delta_obj = new_obj - current_obj
        if delta_obj > 0 or random.random() < np.exp(delta_obj / T):
            current_pos = new_pos
            current_rad = new_rad
            current_obj = new_obj
            current_viol = new_viol
            
            # Update best if valid (or low violation)
            if current_viol < 1e-5:
                if current_obj > best_obj:
                    best_obj = current_obj
                    best_pos = current_pos.copy()
                    best_rad = current_rad.copy()
                    best_viol = current_viol
            elif best_viol > 1e-5 and current_viol < best_viol:
                # If we haven't found a valid one yet, keep track of least violated
                if current_obj > best_obj or current_viol < best_viol:
                     best_obj = current_obj
                     best_pos = current_pos.copy()
                     best_rad = current_rad.copy()
                     best_viol = current_viol
                     
        # Cool down
        T *= alpha
        if T < T_min:
            # Restart with slightly higher T to escape local minima
            T = 0.01
            
    # Final Local Optimization: Grow radii
    # Greedily increase radii as much as possible
    # We do this multiple times in random order
    for _ in range(50):
        order = np.random.permutation(N)
        for k in order:
            # Max possible radius given position
            max_r_wall = min(best_pos[k, 0], L - best_pos[k, 0], best_pos[k, 1], L - best_pos[k, 1])
            
            # Check constraints with other circles
            max_r = max_r_wall
            for j in range(N):
                if j == k: continue
                d = np.linalg.norm(best_pos[k] - best_pos[j])
                # r_k + r_j <= d => r_k <= d - r_j
                # But if we increase r_k, we might violate with j.
                # We want max r_k such that r_k + r_j <= d for all j.
                # So r_k <= min_j (d_kj - r_j)
                # Note: This assumes r_j is fixed.
                constraint = d - best_rad[j]
                if constraint < max_r:
                    max_r = constraint
            
            # Ensure positive
            if max_r > 0:
                best_rad[k] = max_r - 1e-9 # Tiny epsilon for safety
                
    # Verify final result
    # Re-evaluate to ensure no numerical drift made it invalid
    final_obj, final_viol = evaluate(best_pos, best_rad)
    if final_viol > 1e-5:
        # If invalid, fall back to a very safe grid configuration
        # This ensures we always return a valid answer
        side = 5
        cell_size = L / side
        indices = []
        for i in range(side):
            for j in range(side):
                indices.append((i, j))
                if len(indices) == N:
                    break
            if len(indices) == N:
                break
        safe_pos = np.zeros((N, 2))
        safe_rad = np.zeros(N)
        init_r = cell_size * 0.45 # Smaller radius for safety
        for k, (i, j) in enumerate(indices):
            safe_pos[k, 0] = (j + 0.5) * cell_size
            safe_pos[k, 1] = (i + 0.5) * cell_size
            safe_rad[k] = init_r
        return np.column_stack((safe_pos, safe_rad))
        
    return np.column_stack((best_pos, best_rad))
