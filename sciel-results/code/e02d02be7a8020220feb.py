import numpy as np
import time

N = 50
BOX_LIMIT = 0.5

def compute_radius(C):
    # C shape (N, 3)
    # Pairwise distances
    diffs = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    d_min_pair = np.min(dists)
    
    # Wall distances
    wall_dists = BOX_LIMIT - np.abs(C)
    d_min_wall = np.min(wall_dists)
    
    return min(d_min_pair / 2.0, d_min_wall)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Initialization
    # Use a grid to ensure good initial spread
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    C = grid[:N].copy()
    
    # Jitter to break symmetry
    rng = np.random.default_rng(42)
    C += rng.uniform(-0.02, 0.02, size=C.shape)
    C = np.clip(C, -BOX_LIMIT, BOX_LIMIT)
    
    r_best = compute_radius(C)
    C_best = C.copy()
    r_current = r_best
    
    # 2. Simulated Annealing
    T_init = 0.005
    T_min = 1e-5
    cool_rate = 0.98
    steps_per_temp = 50
    step_size_init = 0.02
    step_size_decay = 0.98
    
    T = T_init
    step_size = step_size_init
    
    while time.time() < deadline:
        for _ in range(steps_per_temp):
            if time.time() > deadline:
                break
            
            # Generate move
            delta = rng.normal(0, step_size, size=C.shape)
            C_new = C + delta
            C_new = np.clip(C_new, -BOX_LIMIT, BOX_LIMIT)
            
            r_new = compute_radius(C_new)
            
            if r_new > r_best:
                r_best = r_new
                C_best = C_new.copy()
                
            # Metropolis
            if r_new > r_current:
                C = C_new
                r_current = r_new
            else:
                delta_r = r_new - r_current
                # Avoid division by zero or underflow issues
                if T > 0:
                    prob = np.exp(delta_r / T)
                else:
                    prob = 0.0
                if rng.random() < prob:
                    C = C_new
                    r_current = r_new
        
        T *= cool_rate
        step_size *= step_size_decay
        if T < T_min:
            break
            
    # 3. Local Polish (Force-based)
    # We refine the best solution found
    C = C_best.copy()
    r = r_best
    
    # Iterate to push apart closest pairs and away from walls
    # This is a local search to maximize the minimum distance
    for i in range(1000):
        if time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(C)
        
        # Pairwise repulsion for closest pairs
        diffs = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        d_min = np.min(dists)
        # Threshold for "close" pairs
        eps = 0.05
        close_mask = dists < (d_min + eps)
        
        # Accumulate repulsion
        idx_i, idx_j = np.where(close_mask)
        for ii, jj in zip(idx_i, idx_j):
            if ii == jj: continue
            d = dists[ii, jj]
            if d < 1e-9: continue
            # Force magnitude: stronger if closer
            f_mag = 0.5 * (1 - d / (d_min + eps))
            if f_mag <= 0: continue
            dir_vec = (C[ii] - C[jj]) / d
            forces[ii] += dir_vec * f_mag
            forces[jj] -= dir_vec * f_mag
            
        # Wall attraction
        # If close to wall, push towards center
        wall_eps = 0.05
        for i_idx in range(N):
            for k in range(3):
                dist_to_wall = BOX_LIMIT - np.abs(C[i_idx, k])
                if dist_to_wall < (r + wall_eps):
                    # Push towards 0
                    sign = np.sign(C[i_idx, k])
                    if sign == 0:
                        sign = 1
                    # Strength increases as dist_to_wall decreases
                    strength = 0.5 * (1 - dist_to_wall / (r + wall_eps))
                    if strength > 0:
                        forces[i_idx, k] -= sign * strength
        
        # Update
        # Normalize forces to prevent large jumps
        norm_forces = np.linalg.norm(forces, axis=1)
        max_norm = np.max(norm_forces)
        if max_norm > 1e-9:
            step = 0.1
            C += (forces / max_norm) * step
        else:
            break
            
        C = np.clip(C, -BOX_LIMIT, BOX_LIMIT)
        
        r_new = compute_radius(C)
        if r_new > r:
            r = r_new
            if r > r_best:
                r_best = r
                C_best = C.copy()
        else:
            # If no improvement, we might be stuck in a local minimum of the force field
            # Reduce step size or break?
            # For now, continue, but if many stalls, break.
            pass
            
    return C_best
