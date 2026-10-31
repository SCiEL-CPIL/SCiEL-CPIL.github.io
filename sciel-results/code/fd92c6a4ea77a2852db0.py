"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist

def construct():
    # We want to pack 21 circles in a rectangle with perimeter 4.
    # Perimeter = 2*(w+h) = 4 => w+h = 2.
    # We maximize sum of radii.
    # Assume all circles have the same radius r.
    # Maximize r.
    
    best_result = None
    best_sum = -1
    
    # Grid configurations
    configs = [
        (5, 5),
        (4, 6),
        (6, 4),
        (3, 7),
        (7, 3),
        (2, 11),
        (11, 2),
    ]
    
    # Aspect ratios w, where h = 2 - w
    # We check w in [0.2, 1.0] due to symmetry
    aspect_ratios = np.linspace(0.2, 1.0, 20)
    
    for w_frac in aspect_ratios:
        w = 2 * w_frac
        h = 2 - w
        
        if w < 0.1 or h < 0.1:
            continue
            
        for nx, ny in configs:
            if nx * ny < N:
                continue
                
            # Create grid positions
            dx = w / nx
            dy = h / ny
            
            positions = []
            count = 0
            for i in range(nx):
                for j in range(ny):
                    if count >= N:
                        break
                    x = (i + 0.5) * dx
                    y = (j + 0.5) * dy
                    positions.append([x, y])
                    count += 1
                if count >= N:
                    break
            
            positions = np.array(positions)
            
            # Local Optimization: Gradient Ascent on Min Distance
            # We want to maximize d_min = min(dist(ci, cj), dist(ci, wall))
            
            # Initial jitter
            positions = positions + np.random.uniform(-0.005, 0.005, (N, 2))
            positions[:, 0] = np.clip(positions[:, 0], 0.01, w - 0.01)
            positions[:, 1] = np.clip(positions[:, 1], 0.01, h - 0.01)
            
            best_local_d_min = -1.0
            best_local_positions = positions.copy()
            
            # Parameters
            iterations = 3000
            step_size = 0.02
            cooling = 0.998
            noise_scale = 0.01
            
            for _ in range(iterations):
                # Compute distances
                dists = cdist(positions, positions)
                np.fill_diagonal(dists, np.inf)
                min_pair_dist = np.min(dists)
                
                # Boundary distances
                d_left = positions[:, 0]
                d_right = w - positions[:, 0]
                d_bottom = positions[:, 1]
                d_top = h - positions[:, 1]
                
                min_bound_dist = np.min(np.concatenate([d_left, d_right, d_bottom, d_top]))
                
                d_min = min(min_pair_dist, min_bound_dist)
                
                if d_min > best_local_d_min:
                    best_local_d_min = d_min
                    best_local_positions = positions.copy()
                    
                # Determine gradient direction for the most constrained element
                # If pair is the bottleneck
                if min_pair_dist <= min_bound_dist:
                    i, j = np.unravel_index(np.argmin(dists), dists.shape)
                    vec = positions[i] - positions[j]
                    norm = np.linalg.norm(vec)
                    if norm > 1e-9:
                        vec = vec / norm
                    else:
                        vec = np.array([1.0, 0.0])
                    # Move i away from j, j away from i
                    positions[i] += vec * step_size
                    positions[j] -= vec * step_size
                else:
                    # Boundary is the bottleneck
                    bound_dists = np.column_stack([d_left, d_right, d_bottom, d_top])
                    flat_bound = bound_dists.flatten()
                    min_bound_val = np.min(flat_bound)
                    min_bound_idx = np.argmin(flat_bound)
                    bi = min_bound_idx // 4
                    b_type = min_bound_idx % 4
                    
                    if b_type == 0: # left
                        positions[bi][0] += step_size
                    elif b_type == 1: # right
                        positions[bi][0] -= step_size
                    elif b_type == 2: # bottom
                        positions[bi][1] += step_size
                    elif b_type == 3: # top
                        positions[bi][1] -= step_size
                
                # Add noise
                positions += np.random.normal(0, noise_scale, (N, 2))
                
                # Keep inside
                positions[:, 0] = np.clip(positions[:, 0], 0.0, w)
                positions[:, 1] = np.clip(positions[:, 1], 0.0, h)
                
                step_size *= cooling
                noise_scale *= 0.999
                
            # Final Evaluation
            positions = best_local_positions
            dists = cdist(positions, positions)
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            
            d_left = positions[:, 0]
            d_right = w - positions[:, 0]
            d_bottom = positions[:, 1]
            d_top = h - positions[:, 1]
            min_bound_dist = np.min(np.concatenate([d_left, d_right, d_bottom, d_top]))
            
            d_min = min(min_pair_dist, min_bound_dist)
            r = d_min / 2 - 1e-7
            
            if r <= 0:
                continue
                
            total_sum = N * r
            if total_sum > best_sum:
                best_sum = total_sum
                result = np.zeros((N, 3))
                result[:, 0] = positions[:, 0]
                result[:, 1] = positions[:, 1]
                result[:, 2] = r
                best_result = result

    if best_result is None:
        # Fallback to seed-like solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
