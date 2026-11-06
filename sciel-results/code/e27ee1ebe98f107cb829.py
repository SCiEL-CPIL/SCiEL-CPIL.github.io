import numpy as np
import scipy.optimize
import time
from scipy.spatial.distance import cdist

N = 50

def _score(centers):
    """Compute the radius for a given set of centers."""
    if len(centers) != N:
        return -np.inf
    if not np.all(np.isfinite(centers)):
        return -np.inf
    if np.any(np.abs(centers) > 0.5):
        return -np.inf
        
    dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = np.min(dist_to_boundary)
    
    # Vectorized pair distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)

def _score_fast(centers):
    """Fast score computation using vectorized operations."""
    dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = np.min(dist_to_boundary)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)

def generate_hex_config(n_rows, n_cols, dx, dy, offset_x, offset_y, phase=0):
    """Generate a hexagonal lattice configuration with given parameters."""
    centers = []
    for i in range(n_rows):
        for j in range(n_cols):
            x = offset_x + j * dx + (i % 2) * dx / 2.0
            y = offset_y + i * dy
            centers.append([x, y])
    
    centers = np.array(centers)
    
    # Keep only the first N points
    if len(centers) > N:
        centers = centers[:N]
    
    # If fewer than N, pad with random
    if len(centers) < N:
        extra = np.random.uniform(-0.4, 0.4, (N - len(centers), 2))
        centers = np.vstack([centers, extra])
    
    return centers

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin (evaluator kills at 150s)
    
    best_centers = None
    best_score = -np.inf
    
    # Objective function for optimization
    def objective(x):
        c = x.reshape(-1, 2)
        dist_bound = np.min(0.5 - np.abs(c), axis=1)
        min_bound = np.min(dist_bound)
        
        dists = cdist(c, c)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        r = min(min_pair / 2.0, min_bound)
        return -r
    
    # Strategy 1: Try multiple hexagonal grid configurations and optimize each
    # Based on inspiration: hexagonal lattice with alternating row lengths
    # Joint optimization of spacing, offsets captures higher density
    
    # Try various grid dimensions
    grid_configs = []
    for n_rows in range(5, 10):
        for n_cols in range(5, 10):
            if n_rows * n_cols < N:
                continue
            grid_configs.append((n_rows, n_cols))
    
    # Also try the standard hex spacing
    d_hex = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    
    for n_rows, n_cols in grid_configs:
        if time.time() - start_time > deadline - 5:
            break
        
        # Compute appropriate spacing for this grid
        # For hex packing: dx = 2r, dy = r*sqrt(3)
        # Total width: (n_cols-1)*dx + dx/2 (for offset rows)
        # Total height: (n_rows-1)*dy
        # We want these to fit in [−0.5, 0.5]
        
        # Try a few different spacings
        for dx_factor in [0.9, 0.95, 1.0, 1.05]:
            dx = 1.0 / (n_cols * dx_factor)
            dy = dx * np.sqrt(3.0) / 2.0
            
            # Center the grid
            total_w = (n_cols - 1) * dx + dx / 2.0
            total_h = (n_rows - 1) * dy
            
            offset_x = -total_w / 2.0
            offset_y = -total_h / 2.0
            
            centers = generate_hex_config(n_rows, n_cols, dx, dy, offset_x, offset_y)
            
            # Clip to square
            centers = np.clip(centers, -0.5, 0.5)
            
            s = _score_fast(centers)
            if s > best_score:
                best_score = s
                best_centers = centers.copy()
    
    # Also try the original hex init
    d = np.sqrt(2.0 / (N * np.sqrt(3.0))) * 0.95
    row_height = d * np.sqrt(3.0) / 2.0
    n_rows = int(np.ceil(1.0 / row_height)) + 2
    n_cols = int(np.ceil(1.0 / d)) + 2
    
    centers_list = []
    for i in range(n_rows):
        y = -0.5 + i * row_height
        if i % 2 == 0:
            x_start = -0.5
        else:
            x_start = -0.5 + d / 2.0
        for j in range(n_cols):
            x = x_start + j * d
            centers_list.append([x, y])
    
    centers_arr = np.array(centers_list)
    centers_arr = np.clip(centers_arr, -0.5, 0.5)
    
    if len(centers_arr) > N:
        dist_to_boundary = np.min(0.5 - np.abs(centers_arr), axis=1)
        sort_idx = np.argsort(dist_to_boundary)
        centers_arr = centers_arr[sort_idx[-N:]]
    elif len(centers_arr) < N:
        extra = np.random.uniform(-0.4, 0.4, (N - len(centers_arr), 2))
        centers_arr = np.vstack([centers_arr, extra])
    
    s = _score_fast(centers_arr)
    if s > best_score:
        best_score = s
        best_centers = centers_arr.copy()
    
    # If no valid configuration found, fallback
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Strategy 2: Local optimization using Powell
    if time.time() - start_time < deadline - 10:
        try:
            res = scipy.optimize.minimize(
                objective,
                best_centers.flatten(),
                method='Powell',
                options={'maxiter': 300, 'disp': False, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            opt_centers = res.x.reshape(-1, 2)
            opt_score = -res.fun
            
            if opt_score > best_score:
                best_score = opt_score
                best_centers = opt_centers.copy()
        except Exception:
            pass
    
    # Strategy 3: Nelder-Mead polish
    if time.time() - start_time < deadline - 5:
        try:
            res2 = scipy.optimize.minimize(
                objective,
                best_centers.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'disp': False, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if -res2.fun > best_score:
                best_score = -res2.fun
                best_centers = res2.x.reshape(-1, 2)
        except Exception:
            pass
    
    # Strategy 4: Random perturbation local search (from inspiration 2)
    if time.time() - start_time < deadline - 3:
        centers = best_centers.copy()
        step_size = 0.01
        patience = 0
        max_patience = 30
        local_best = best_score
        
        for it in range(2000):
            if time.time() - start_time > deadline - 1:
                break
            
            if patience > max_patience:
                centers = best_centers.copy()
                step_size *= 1.3
                patience = 0
            
            noise = np.random.randn(N, 2) * step_size
            new_centers = centers + noise
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            new_score = _score_fast(new_centers)
            
            if new_score > local_best:
                local_best = new_score
                best_score = new_score
                best_centers = new_centers.copy()
                centers = new_centers.copy()
                patience = 0
                step_size *= 0.95
            else:
                patience += 1
                step_size *= 0.92
    
    # Final clip and return
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
