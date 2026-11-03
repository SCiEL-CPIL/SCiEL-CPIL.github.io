"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

def _compute_radius(centers):
    """Compute the maximum common radius for given centers in [-0.5, 0.5]^3."""
    # Distance to walls
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(np.min(wall_dists), min_pair_dist / 2.0)

def _neg_radius(centers):
    """Negative radius for minimization."""
    return -_compute_radius(centers)

def _gradient_neg_radius(centers):
    """Gradient of negative radius (approximate via finite differences or analytic)."""
    # Use subgradient approach: identify which constraint is active
    wall_dists = 0.5 - np.abs(centers)
    min_wall_idx = np.argmin(wall_dists, axis=1)
    min_wall_val = np.min(wall_dists, axis=1)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_idx = np.argmin(dists)
    i, j = np.unravel_index(min_pair_idx, (N, N))
    min_pair_val = dists[i, j]
    
    r = min(np.min(min_wall_val), min_pair_val / 2.0)
    
    grad = np.zeros_like(centers)
    
    if np.min(min_wall_val) <= min_pair_val / 2.0:
        # Wall constraint is active
        for i in range(N):
            if min_wall_val[i] == np.min(min_wall_val):
                # Gradient of wall distance
                if centers[i, min_wall_idx[i]] > 0:
                    grad[i, min_wall_idx[i]] = 1.0
                else:
                    grad[i, min_wall_idx[i]] = -1.0
    else:
        # Pair constraint is active
        # d/dci (|ci - cj|/2) = (ci - cj) / |ci - cj|
        # d/dcj (|ci - cj|/2) = (cj - ci) / |ci - cj|
        vec = centers[i] - centers[j]
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            grad[i] = vec / norm
            grad[j] = -vec / norm
    
    return -grad  # Negative because we're minimizing -r

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial guess: simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Use Nelder-Mead optimization
    for attempt in range(5):
        if time.time() > deadline:
            break
        
        # Add small random perturbation for restart
        if attempt > 0:
            centers = best_centers + np.random.randn(N, 3) * 0.01
            # Clip to stay in bounds
            centers = np.clip(centers, -0.49, 0.49)
        
        # Optimize
        try:
            result = minimize(
                _neg_radius,
                centers,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.fun < -best_radius:
                new_radius = -result.fun
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = result.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                _neg_radius,
                best_centers,
                method='Nelder-Mead',
                options={
                    'maxiter': 20000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if result.fun < -best_radius:
                best_radius = -result.fun
                best_centers = result.x.copy()
        except Exception:
            pass
    
    # Ensure validity: clip to cube and check
    best_centers = np.clip(best_centers, -0.499999, 0.499999)
    
    return best_centers
# EVOLVE-BLOCK-END
