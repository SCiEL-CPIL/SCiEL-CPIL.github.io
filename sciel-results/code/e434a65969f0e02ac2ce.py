"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START

def _initial_grid():
    """Create an initial grid arrangement of N points in [-0.5, 0.5]^2."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = np.array([(x, y) for y in ticks for x in ticks])
    return pts[:N].astype(np.float64)

def _compute_radius(pts):
    """Compute the maximum common radius for given centers."""
    if len(pts) < 2:
        return 0.0
    from scipy.spatial.distance import cdist
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    min_boundary = np.min(np.minimum(dx, dy))
    return min(min_dist / 2.0, min_boundary)

def _smooth_objective(x, beta=100.0):
    """Smooth objective: maximize min distance using log-sum-exp approximation."""
    pts = x.reshape(-1, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Log-sum-exp of -dist_sq/beta to approximate -min_dist
    # We want to maximize min_dist, so minimize -min_dist
    # Use: -min(dist) ≈ (1/beta) * log(sum(exp(-beta*dist_sq)))
    # But we need to be careful with numerical stability
    
    # Distance to boundary
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    boundary_dist = np.minimum(dx, dy)
    
    # Penalty for being outside
    boundary_violation = np.sum(np.maximum(0, -boundary_dist))
    
    # Smooth approximation of -min_pair_dist
    # Use log-sum-exp trick for numerical stability
    min_dist_sq = np.min(dist_sq, axis=(0,1))
    
    # Combine: we want to maximize min(min_pair_dist, min_boundary)
    # Approximate with smooth function
    r_approx = np.minimum(np.sqrt(min_dist_sq), np.min(boundary_dist))
    
    # Penalty for violations
    penalty = boundary_violation * 1000.0
    
    return -r_approx + penalty

def _local_search(pts, rng, time_budget=30.0):
    """Local search with random perturbations and acceptance."""
    start_time = time.time()
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    step_size = 0.02
    
    while time.time() - start_time < time_budget:
        # Generate random perturbation
        delta = rng.normal(0, step_size, size=best_pts.shape)
        new_pts = best_pts + delta
        new_pts = np.clip(new_pts, -0.5, 0.5)
        
        new_r = _compute_radius(new_pts)
        
        if new_r > best_r:
            best_r = new_r
            best_pts = new_pts
            step_size = max(0.001, step_size * 0.95)
        else:
            step_size = min(0.05, step_size * 1.05)
    
    return best_pts, best_r

def _gradient_optimize(pts, time_budget=30.0):
    """Use L-BFGS-B with smooth objective."""
    start_time = time.time()
    x0 = pts.flatten().astype(np.float64)
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    res = minimize(_smooth_objective, x0, method='L-BFGS-B', 
                   bounds=bounds,
                   options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
    
    new_pts = res.x.reshape(-1, 2)
    new_pts = np.clip(new_pts, -0.5, 0.5)
    
    return new_pts

def construct():
    """Construct centers of N equal circles inside the square [-1/2, 1/2]^2."""
    start_time = time.time()
    time_limit = 140.0
    
    # Start with grid
    pts = _initial_grid()
    best_radius = _compute_radius(pts)
    best_pts = pts.copy()
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Gradient optimization from grid
    try:
        if time.time() - start_time < time_limit - 40:
            grad_pts = _gradient_optimize(pts, time_budget=30.0)
            grad_r = _compute_radius(grad_pts)
            if grad_r > best_radius:
                best_radius = grad_r
                best_pts = grad_pts
    except Exception:
        pass
    
    # Strategy 2: Local search from best point
    try:
        if time.time() - start_time < time_limit - 30:
            ls_pts, ls_r = _local_search(best_pts, rng, time_budget=25.0)
            if ls_r > best_radius:
                best_radius = ls_r
                best_pts = ls_pts
    except Exception:
        pass
    
    # Strategy 3: Multiple random restarts with local search
    try:
        remaining_time = time_limit - (time.time() - start_time) - 10
        if remaining_time > 10:
            n_restarts = int(remaining_time / 5)
            n_restarts = max(1, min(n_restarts, 10))
            
            for i in range(n_restarts):
                if time.time() - start_time > time_limit - 10:
                    break
                
                # Random initial points
                rand_pts = rng.uniform(-0.45, 0.45, size=(N, 2))
                rand_pts = np.clip(rand_pts, -0.5, 0.5)
                
                # Quick local search
                ls_pts, ls_r = _local_search(rand_pts, rng, time_budget=3.0)
                
                if ls_r > best_radius:
                    best_radius = ls_r
                    best_pts = ls_pts
    except Exception:
        pass
    
    # Final local search polish
    try:
        if time.time() - start_time < time_limit - 5:
            final_pts, final_r = _local_search(best_pts, rng, time_budget=5.0)
            if final_r > best_radius:
                best_radius = final_r
                best_pts = final_pts
    except Exception:
        pass
    
    return best_pts

# EVOLVE-BLOCK-END
