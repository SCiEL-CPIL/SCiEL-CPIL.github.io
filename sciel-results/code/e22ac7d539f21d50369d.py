import numpy as np
import time
from scipy.optimize import minimize

N = 100

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

def _bcs_initial_grid():
    """
    Generate initial points using a body-centered cubic (BCC) lattice pattern.
    This provides a denser starting configuration than simple cubic grid.
    """
    points = []
    
    # Use a 5x4x5 grid of BCC unit cells
    # Each BCC cell has 1 corner point + 1 center point
    # But we need to be careful about boundaries
    
    x_ticks = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    y_ticks = np.array([-0.375, -0.125, 0.125, 0.375])
    z_ticks = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    
    # Simple grid first (5*4*5 = 100 points)
    for x in x_ticks:
        for y in y_ticks:
            for z in z_ticks:
                points.append([x, y, z])
                if len(points) >= N:
                    break
            if len(points) >= N:
                break
        if len(points) >= N:
            break
    
    return np.array(points[:N])

def _simulated_annealing(x0, max_time=60.0, start_temp=0.05, end_temp=0.001, 
                          cooling_rate=0.999, steps_per_temp=50):
    """
    Simulated annealing to improve the packing.
    """
    start_time = time.time()
    x = x0.copy()
    best_x = x.copy()
    best_score = _calc_score(x)
    
    current_temp = start_temp
    
    while time.time() < start_time + max_time and current_temp > end_temp:
        # Generate a random perturbation
        num_moves = np.random.randint(1, 5)
        indices = np.random.choice(N, num_moves, replace=False)
        
        # Random displacement
        displacement = np.random.normal(0, current_temp, size=(num_moves, 3))
        
        # Try the move
        x_new = x.copy()
        x_new[indices] += displacement
        x_new = np.clip(x_new, -0.499, 0.499)
        
        # Calculate score difference
        score_new = _calc_score(x_new)
        score_diff = score_new - best_score
        
        # Accept if better or with some probability
        if score_diff > 0:
            x = x_new
            if score_new > best_score:
                best_score = score_new
                best_x = x_new.copy()
        else:
            # Boltzmann acceptance
            if np.random.rand() < np.exp(score_diff / (current_temp * 0.01)):
                x = x_new
        
        # Cool down
        current_temp *= cooling_rate
    
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for safety
    
    # Generate initial grid
    base_grid = _bcs_initial_grid()
    best_x = base_grid.copy()
    best_score = _calc_score(best_x)
    
    # Objective function for scipy
    def objective(x_flat):
        e, g = _energy_and_grad(x_flat)
        return e, g
        
    bounds = [(-0.5, 0.5)] * 3 * N
    
    np.random.seed(42)
    
    # Phase 1: Local optimization with different jitter levels
    jitter_levels = [0.0, 0.005, 0.01, 0.02, 0.05]
    
    for jitter in jitter_levels:
        if time.time() > deadline - 30:
            break
            
        # Create initial guess
        if jitter == 0.0:
            x0 = best_x.copy()
        else:
            x0 = best_x + np.random.uniform(-jitter, jitter, best_x.shape)
            x0 = np.clip(x0, -0.49, 0.49)
            
        # Run optimization
        try:
            res = minimize(objective, x0.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10})
            
            if res.success:
                new_x = res.x.reshape(N, 3)
                new_x = np.clip(new_x, -0.5, 0.5)
                new_score = _calc_score(new_x)
                
                if new_score > best_score:
                    best_score = new_score
                    best_x = new_x
                    
        except Exception:
            pass
    
    # Phase 2: Simulated annealing for global improvement
    if time.time() < deadline - 15:
        try:
            sa_time = min(30.0, deadline - time.time() - 5)
            if sa_time > 5:
                sa_x, sa_score = _simulated_annealing(
                    best_x, max_time=sa_time, start_temp=0.02, end_temp=0.0005,
                    cooling_rate=0.9995, steps_per_temp=30
                )
                if sa_score > best_score:
                    best_score = sa_score
                    best_x = sa_x
        except Exception:
            pass
    
    # Phase 3: Final fine-tuning
    if time.time() < deadline - 5:
        try:
            res_final = minimize(objective, best_x.flatten(), method='L-BFGS-B', jac=True,
                                 bounds=bounds,
                                 options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12})
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
