"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time


def _compute_score(centers):
    """Compute the minimum radius for a given set of centers."""
    # Check boundaries
    min_boundary = np.min(np.minimum(
        np.minimum(0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1]))
    ))
    
    # Check pairwise distances
    if len(centers) < 2:
        return min_boundary
    
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_boundary, min_pair_dist / 2)


def _objective(params):
    """Objective function to minimize (negative radius)."""
    centers = params.reshape(-1, 2)
    r = _compute_score(centers)
    return -r


def _gradient(params):
    """Numerical gradient."""
    eps = 1e-7
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _objective(params_plus)
        
        params_minus = params.copy()
        params_minus[i] -= eps
        f_minus = _objective(params_minus)
        
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def construct():
    """Construct centers of 50 equal circles in the unit square."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with a grid arrangement
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    best_centers = centers.copy()
    best_score = _compute_score(centers)
    
    # Try multiple random restarts
    np.random.seed(42)
    num_restarts = 10
    restarts_done = 0
    
    while restarts_done < num_restarts:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Create a perturbed starting point
        if restarts_done == 0:
            current_centers = centers.copy()
        else:
            # Random initialization within the square
            current_centers = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        # Optimize using L-BFGS-B with bounds
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        result = minimize(
            _objective,
            current_centers.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'maxfun': 10000, 'gtol': 1e-8}
        )
        
        new_centers = result.x.reshape(-1, 2)
        new_score = _compute_score(new_centers)
        
        if new_score > best_score:
            best_score = new_score
            best_centers = new_centers.copy()
        
        restarts_done += 1
    
    # Final polish with tighter tolerance
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        result = minimize(
            _objective,
            best_centers.flatten(),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 2000, 'maxfun': 20000, 'gtol': 1e-10}
        )
        polished_centers = result.x.reshape(-1, 2)
        polished_score = _compute_score(polished_centers)
        
        if polished_score > best_score:
            best_score = polished_score
            best_centers = polished_centers.copy()
    
    return best_centers
# EVOLVE-BLOCK-END
