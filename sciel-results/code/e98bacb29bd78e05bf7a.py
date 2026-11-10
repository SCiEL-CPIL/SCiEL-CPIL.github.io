"""Improved packing of 50 equal spheres in a unit cube using Vectorized Soft-Min Optimization."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _exact_score(centers):
    """Exact minimum radius score."""
    if len(centers) < 2:
        return 0.5
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    min_wall = np.min(wall_dists)
    
    # Vectorized pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_pair / 2.0, min_wall)

def _soft_min_score_vec(centers, beta=50.0):
    """Differentiable soft-minimum score for gradient-based optimization using vectorized ops."""
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    min_wall = np.min(wall_dists)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    a = min_pair / 2.0
    b = min_wall
    
    # Soft min: -1/beta * log(exp(-beta*a) + exp(-beta*b))
    m = min(a, b)
    # Numerical stability
    val = m - math.log(math.exp(-beta * (a - m)) + math.exp(-beta * (b - m))) / beta
    return val

def _objective_vec(x, beta=50.0):
    """Objective for L-BFGS-B: minimize negative soft-min score."""
    centers = x.reshape(N, 3)
    centers = np.clip(centers, -0.5, 0.5)
    return -_soft_min_score_vec(centers, beta=beta)

def _initial_grid():
    """4x4x4 grid, first 50 points."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _optimize_sequence(centers, time_limit=30.0):
    """
    Run a sequence of L-BFGS-B optimizations with increasing beta.
    """
    start_time = time.time()
    current_centers = centers.copy()
    
    # Beta schedule: start smooth, end sharp
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    for beta in betas:
        if time.time() - start_time > time_limit * 0.8:
            break
            
        x0 = current_centers.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            result = minimize(
                _objective_vec, x0, args=(beta,),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            current_centers = np.clip(result.x.reshape(N, 3), -0.5, 0.5)
        except Exception:
            break
            
    return current_centers

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    # 1. Initial Grid
    centers = _initial_grid()
    best_centers = centers.copy()
    best_score = _exact_score(best_centers)
    
    # 2. Multi-start Optimization
    # We will try the base grid and a few random perturbations
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Perturb the grid slightly to break symmetry
        if r == 0:
            init_centers = centers.copy()
        else:
            # Random perturbation of the grid
            perturbation = np.random.uniform(-0.01, 0.01, (N, 3))
            init_centers = np.clip(centers + perturbation, -0.5, 0.5)
            
        # Allocate time per restart
        time_per_restart = min(20.0, (deadline - 10.0 - (time.time() - start_time)) / max(1, num_restarts - r))
        
        optimized_centers = _optimize_sequence(init_centers, time_limit=time_per_restart)
        
        score = _exact_score(optimized_centers)
        if score > best_score:
            best_score = score
            best_centers = optimized_centers
            
    # 3. Final Polish
    # Use the best centers found and run a very high beta optimization
    if time.time() < deadline - 5.0:
        final_centers = _optimize_sequence(best_centers, time_limit=5.0)
        final_score = _exact_score(final_centers)
        if final_score > best_score:
            best_score = final_score
            best_centers = final_centers
            
    return best_centers
# EVOLVE-BLOCK-END
