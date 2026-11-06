"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    def get_radius(c):
        # Boundary constraint
        boundary_min = np.min(0.5 - np.abs(c))
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists)
        
        return min(boundary_min, pair_min / 2.0)

    # Initialize with a grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random noise to break symmetry
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.01, centers.shape)
    
    # Clip to box
    centers = np.clip(centers, -0.5, 0.5)
    
    current_radius = get_radius(centers)
    best_centers = centers.copy()
    best_radius = current_radius
    
    # SA Parameters
    T = 0.01
    T_min = 1e-7
    cooling_rate = 0.9999
    step_scale = 0.02
    
    steps_since_improvement = 0
    max_steps_without_improvement = 2000
    
    while time.time() < deadline:
        i = rng.integers(0, N)
        
        # Adaptive step size? 
        # Let's keep it simple first.
        disp = rng.normal(0, step_scale, size=3)
        new_c = centers[i] + disp
        
        if np.any(np.abs(new_c) > 0.5):
            continue
            
        # Calculate new radius components
        others = np.concatenate([centers[:i], centers[i+1:]])
        diff = new_c - others
        dists = np.linalg.norm(diff, axis=1)
        min_dist_to_others = np.min(dists)
        
        # Min dist among others
        if N > 2:
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            sub_centers = centers[mask]
            diff_sub = sub_centers[:, np.newaxis, :] - sub_centers[np.newaxis, :, :]
            dists_sub = np.linalg.norm(diff_sub, axis=2)
            np.fill_diagonal(dists_sub, np.inf)
            min_dist_rest = np.min(dists_sub)
        else:
            min_dist_rest = np.inf
            
        new_bound = 0.5 - np.min(np.abs(new_c))
        new_radius = min(new_bound, min_dist_to_others / 2.0, min_dist_rest / 2.0)
        
        delta_r = new_radius - current_radius
        if delta_r > 0:
            centers[i] = new_c
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
        else:
            prob = np.exp(delta_r / T)
            if rng.random() < prob:
                centers[i] = new_c
                current_radius = new_radius
                steps_since_improvement += 1
            else:
                steps_since_improvement += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Restart logic
        if steps_since_improvement > max_steps_without_improvement:
            centers = best_centers.copy()
            current_radius = best_radius
            T = 0.01  # Reheat
            steps_since_improvement = 0
            
    return best_centers
# EVOLVE-BLOCK-END
