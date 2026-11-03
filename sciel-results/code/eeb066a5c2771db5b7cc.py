import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    def generate_initial_points():
        # Hexagonal lattice initialization
        spacing = 0.1
        while True:
            pts = []
            j_range = int(1.2 / (spacing * np.sqrt(3)/2)) + 2
            i_range = int(1.2 / spacing) + 2
            for j in range(-j_range, j_range+1):
                for i in range(-i_range, i_range+1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    if np.hypot(x, y) <= 0.99:
                        pts.append([x, y])
            pts = np.array(pts)
            if len(pts) >= N:
                # Randomly select N points to break symmetry
                idx = np.random.choice(len(pts), N, replace=False)
                return pts[idx]
            spacing *= 0.95
            if spacing < 0.01: break
        
        # Fallback: Random points
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N)) * 0.99
        return np.c_[radii * np.cos(angles), radii * np.sin(angles)]

    c = generate_initial_points()
    
    step_size = 0.05
    best_score = -np.inf
    best_c = c.copy()
    
    for iter in range(50000):
        if time.time() > deadline:
            break
            
        # Compute distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        dists_bound = 1.0 - np.linalg.norm(c, axis=1)
        
        min_pair_dist = np.min(dists)
        min_bound_dist = np.min(dists_bound)
        current_score = min(min_pair_dist / 2.0, min_bound_dist)
        
        if current_score > best_score:
            best_score = current_score
            best_c = c.copy()
            
        # Find critical pairs (those close to the minimum distance)
        threshold = min_pair_dist + 1e-5
        mask = dists < threshold
        mask.fill_diagonal(False)
        
        forces = np.zeros_like(c)
        
        # Accumulate repulsive forces from critical pairs
        i_idx, j_idx = np.where(mask)
        for i, j in zip(i_idx, j_idx):
            vec = c[i] - c[j]
            forces[i] += vec
            forces[j] += -vec
            
        # Accumulate boundary forces
        bound_threshold = min_bound_dist + 1e-5
        bound_mask = dists_bound < bound_threshold
        for i in np.where(bound_mask)[0]:
            # Push towards center
            forces[i] += -c[i]
            
        # Normalize forces and apply step
        norms = np.linalg.norm(forces, axis=1)
        norms[norms == 0] = 1.0
        c += (forces / norms[:, np.newaxis]) * step_size
        
        # Project back to unit circle
        norms_c = np.linalg.norm(c, axis=1)
        mask_out = norms_c > 1.0
        if np.any(mask_out):
            c[mask_out] = c[mask_out] / norms_c[mask_out, np.newaxis]
            
        # Decay step size
        step_size *= 0.9995
        
    return best_c
