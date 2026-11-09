"""Program for packing 50 equal circles in a unit square."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Find centers of 50 equal non-overlapping circles in [-0.5, 0.5]^2
    maximizing the minimum distance to walls and between circles.
    
    Score = min(
        min_{i<j} ||c_i - c_j|| / 2,
        min_i min_k (0.5 - |c_{ik}|)
    )
    """
    # Helper: create hexagonal layout
    def hexagonal_layout(n):
        """Create hexagonal packing layout."""
        rows = 7
        centers = []
        row_height = 0.15
        col_width = 0.14
        offset = 0.07
        
        for i in range(rows):
            cols = 7
            for j in range(cols):
                x = (j - 3) * col_width + (offset if i % 2 == 1 else 0)
                y = (i - 3) * row_height
                centers.append([x, y])
        
        while len(centers) < n:
            centers.append([0.0, 0.0])
        
        centers = np.array(centers[:n])
        max_abs = np.max(np.abs(centers))
        if max_abs > 0:
            centers = centers * (0.45 / max_abs)
        
        return centers
    
    # Objective with mode selection
    def objective(x, mode='combined'):
        """Return negative of the minimum radius (to minimize)."""
        centers = x.reshape(N, 2)
        
        wall_dists = np.min(np.stack([
            0.5 - np.abs(centers[:, 0]),
            0.5 - np.abs(centers[:, 1])
        ]), axis=1)
        min_wall = np.min(wall_dists)
        
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        if mode == 'pair':
            r = min_pair
        elif mode == 'wall':
            r = min_wall
        else:
            r = min(min_wall, min_pair)
        
        return -r
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    np.random.seed(42)
    
    # Generate initial guesses
    x_hex = hexagonal_layout(N).flatten()
    
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    x_grid = grid[:N].flatten()
    
    x_random1 = np.random.uniform(-0.4, 0.4, size=2 * N)
    x_random2 = np.random.uniform(-0.4, 0.4, size=2 * N)
    
    initial_guesses = [x_hex, x_grid, x_random1, x_random2]
    
    best_x = None
    best_score = np.inf
    
    # Phase 1: Optimize pairwise distance first
    for x_start in initial_guesses:
        x_start = np.clip(x_start, -0.49, 0.49)
        
        try:
            result = minimize(
                lambda x: objective(x, mode='pair'),
                x_start,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-8}
            )
            
            if not np.isnan(result.fun):
                x_temp = result.x
                score_temp = objective(x_temp, mode='combined')
                
                if score_temp < best_score:
                    best_score = score_temp
                    best_x = x_temp.copy()
        except Exception:
            continue
    
    # Phase 2: Refine with combined objective
    if best_x is not None:
        for refinement in range(3):
            try:
                result = minimize(
                    lambda x: objective(x, mode='combined'),
                    best_x,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-10}
                )
                
                if not np.isnan(result.fun) and result.fun < best_score:
                    best_score = result.fun
                    best_x = result.x.copy()
            except Exception:
                break
    
    # Phase 3: Random restarts near best solution
    for trial in range(10):
        if best_x is None:
            break
        
        x_start = best_x + np.random.randn(2 * N) * 0.005
        x_start = np.clip(x_start, -0.49, 0.49)
        
        try:
            result = minimize(
                lambda x: objective(x, mode='combined'),
                x_start,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-10}
            )
            
            if not np.isnan(result.fun) and result.fun < best_score:
                best_score = result.fun
                best_x = result.x.copy()
        except Exception:
            continue
    
    # Final tight refinement
    if best_x is not None:
        try:
            result = minimize(
                lambda x: objective(x, mode='combined'),
                best_x,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            
            if not np.isnan(result.fun) and result.fun < best_score:
                best_score = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    centers = best_x.reshape(N, 2)
    centers = np.clip(centers, -0.4999, 0.4999)
    
    return centers

# EVOLVE-BLOCK-END
