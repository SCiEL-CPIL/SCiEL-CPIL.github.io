# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    # 1. Initial Keich construction
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)
        
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = keich[i_flat]
    base_y = keich[j_flat]
    positions = np.stack([base_x, base_y], axis=1).copy()
    
    # Precompute indices
    indices = np.arange(N * N)
    i_idx = indices // N
    j_idx = indices % N
    
    # Fast Volume Calculator for Optimization
    def compute_volume_fast(pos):
        # 11 points for Simpson's rule (5 intervals)
        c_vals = np.linspace(0, 1, 11)
        areas = np.zeros(len(c_vals))
        
        for idx_c, c in enumerate(c_vals):
            x_min = pos[:, 0] + c * i_idx / 8.0
            x_max = x_min + 0.125
            y_min = pos[:, 1] + c * j_idx / 8.0
            y_max = y_min + 0.125
            
            x_coords = np.unique(np.concatenate((x_min, x_max)))
            if len(x_coords) < 2:
                areas[idx_c] = 0.0
                continue
                
            area = 0.0
            for m in range(len(x_coords) - 1):
                x_left = x_coords[m]
                x_right = x_coords[m + 1]
                width = x_right - x_left
                if width <= 0:
                    continue
                mask = (x_min <= x_left) & (x_max >= x_right)
                if not np.any(mask):
                    continue
                y_mins = y_min[mask]
                y_maxs = y_max[mask]
                order = np.argsort(y_mins)
                y_mins_sorted = y_mins[order]
                y_maxs_sorted = y_maxs[order]
                
                current_max = -np.inf
                y_union_len = 0.0
                for k in range(len(y_mins_sorted)):
                    if y_mins_sorted[k] > current_max:
                        y_union_len += y_maxs_sorted[k] - y_mins_sorted[k]
                        current_max = y_maxs_sorted[k]
                    else:
                        if y_maxs_sorted[k] > current_max:
                            y_union_len += y_maxs_sorted[k] - current_max
                            current_max = y_maxs_sorted[k]
                area += width * y_union_len
            areas[idx_c] = area
            
        # Simpson's Rule for 11 points (5 intervals)
        # h = 0.2
        # Integral = h/3 * (f0 + f10 + 4(f1+f3+f5+f7+f9) + 2(f2+f4+f6+f8))
        h = 0.2
        vol = (h / 3.0) * (areas[0] + areas[-1] + 
                           4.0 * np.sum(areas[1:-1:2]) + 
                           2.0 * np.sum(areas[2:-1:2]))
        return vol

    start_time = time.time()
    time_limit = 580.0
    
    best_pos = positions.copy()
    best_vol = compute_volume_fast(best_pos)
    
    def local_search(pos, max_iter=10):
        pos = pos.copy()
        curr_vol = compute_volume_fast(pos)
        step_size = 0.05
        for i in range(max_iter):
            improved = False
            coords = np.random.permutation(128)
            for c_idx in coords:
                k = c_idx // 2
                axis = c_idx % 2
                orig_val = pos[k, axis]
                # Try + and - step
                pos[k, axis] = orig_val + step_size
                v1 = compute_volume_fast(pos)
                pos[k, axis] = orig_val - step_size
                v2 = compute_volume_fast(pos)
                
                if v1 < curr_vol:
                    pos[k, axis] = orig_val + step_size
                    curr_vol = v1
                    improved = True
                elif v2 < curr_vol:
                    pos[k, axis] = orig_val - step_size
                    curr_vol = v2
                    improved = True
                else:
                    pos[k, axis] = orig_val
            if not improved:
                # Try larger step if stuck?
                step_size *= 0.5
                if step_size < 0.001:
                    break
        return pos, curr_vol

    # Run initial local search
    pos, vol = local_search(best_pos, max_iter=20)
    if vol < best_vol:
        best_vol = vol
        best_pos = pos.copy()
        
    # Random restarts
    while time.time() - start_time < time_limit:
        # Perturb
        perturb_pos = best_pos + np.random.normal(0, 0.1, size=best_pos.shape)
        pos, vol = local_search(perturb_pos, max_iter=10)
        if vol < best_vol:
            best_vol = vol
            best_pos = pos.copy()
            
    return best_pos

# EVOLVE-BLOCK-END
