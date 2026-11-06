import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _energy_and_grad(x):
    """
    x: shape (N, 3)
    Returns energy and gradient.
    Energy: Sum of inverse square distances between all pairs + inverse square distances to walls.
    """
    x = x.reshape(N, 3)
    
    # Compute pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero
    mask = ~np.eye(N, dtype=bool)
    dist_sq_masked = np.where(mask, dist_sq, 1.0)
    
    # Energy from pairs
    e_pairs = np.sum(1.0 / dist_sq_masked)
    
    # Gradient of E_pairs
    factor = 1.0 / (dist_sq_masked**2)
    factor[np.eye(N, dtype=bool)] = 0.0
    weighted_diff = diff * factor[:, :, np.newaxis]
    grad_pairs = -2.0 * np.sum(weighted_diff, axis=1)
    
    # Wall distances
    d_left = x + 0.5
    d_right = 0.5 - x
    
    eps = 1e-6
    d_left = np.maximum(d_left, eps)
    d_right = np.maximum(d_right, eps)
    
    e_walls = np.sum(1.0 / (d_left**2)) + np.sum(1.0 / (d_right**2))
    
    # Gradient of E_walls
    grad_walls = -2.0 / (d_left**3) + 2.0 / (d_right**3)
    
    total_energy = e_pairs + e_walls
    total_grad = grad_pairs + grad_walls
    
    return total_energy, total_grad.flatten()

def _calc_score(pts):
    """Calculate the actual score (radius) for a set of points."""
    if pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
        
    # Pair distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1e10)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    # Wall distances
    min_wall_dist = np.min(0.5 - np.abs(pts))
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for safety
    
    # Base grid: 5x4x5
    x_ticks = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    y_ticks = np.array([-0.375, -0.125, 0.125, 0.375])
    z_ticks = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    
    grid = np.array([(x, y, z) for x in x_ticks for y in y_ticks for z in z_ticks])
    base_grid = grid[:N]
    
    best_x = base_grid.copy()
    best_score = _calc_score(best_x)
    
    # Objective function for scipy
    def objective(x_flat):
        e, g = _energy_and_grad(x_flat)
        return e, g
        
    bounds = [(-0.5, 0.5)] * 3 * N
    
    # Strategy: Multiple restarts with different jitter levels
    # The grid is symmetric, so we need to break symmetry to allow movement.
    
    np.random.seed(42)
    
    jitter_levels = [0.0, 0.01, 0.02, 0.05, 0.1]
    
    for jitter in jitter_levels:
        if time.time() > deadline - 20:
            break
            
        # Create initial guess
        if jitter == 0.0:
            x0 = base_grid.copy()
        else:
            x0 = base_grid + np.random.uniform(-jitter, jitter, base_grid.shape)
            x0 = np.clip(x0, -0.49, 0.49)
            
        # Run optimization
        try:
            # Use L-BFGS-B
            # We limit maxiter to avoid hanging
            res = minimize(objective, x0.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.success:
                new_x = res.x.reshape(N, 3)
                # Clip to ensure validity
                new_x = np.clip(new_x, -0.5, 0.5)
                new_score = _calc_score(new_x)
                
                if new_score > best_score:
                    best_score = new_score
                    best_x = new_x
                    
        except Exception as e:
            pass
            
    # If we have time, do a final fine-tuning on the best solution
    if time.time() < deadline - 10:
        try:
            res_final = minimize(objective, best_x.flatten(), method='L-BFGS-B', jac=True,
                                 bounds=bounds,
                                 options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14})
            if res_final.success:
                final_x = res_final.x.reshape(N, 3)
                final_x = np.clip(final_x, -0.5, 0.5)
                final_score = _calc_score(final_x)
                if final_score > best_score:
                    best_score = final_score
                    best_x = final_x
        except Exception:
            pass

    return best_x
