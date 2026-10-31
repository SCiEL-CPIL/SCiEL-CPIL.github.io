"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

def _score(points):
    """Compute (max_dist / min_dist)^2 for a set of points."""
    n = len(points)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Exclude self-distances (diagonal)
    mask = ~np.eye(n, dtype=bool)
    pairwise_dists = dists[mask]
    if len(pairwise_dists) == 0:
        return np.inf
    min_d = np.min(pairwise_dists)
    max_d = np.max(pairwise_dists)
    if min_d < 1e-12:
        return np.inf
    return (max_d / min_d) ** 2

def construct():
    """
    Place 16 distinct points in 2D to minimize the squared ratio of
    maximum to minimum pairwise distance.
    
    Strategy:
    1. Start with a 4x4 grid arrangement (known good baseline).
    2. Use differential evolution to optimize the positions.
    3. Polish with L-BFGS-B for local refinement.
    4. Enforce a deadline guard.
    """
    deadline = time.time() + 120  # 120s safety margin
    
    # Initial configuration: 4x4 grid
    side = 4
    grid = np.array([(i, j) for i in range(side) for j in range(side)], dtype=float)
    # Center and scale
    grid -= grid.mean(axis=0)
    grid *= 1.0  # unit scale initially
    
    best_points = grid.copy()
    best_score = _score(best_points)
    
    # Define objective function for optimization
    def objective(x):
        pts = x.reshape(N, 2)
        s = _score(pts)
        return s
    
    # Use differential evolution for global optimization
    # Bounds: points in [-2, 2]^2 (we can scale later)
    bounds = [(-2, 2)] * (N * 2)
    
    x0 = best_points.flatten()
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=200,
            tol=1e-8,
            polish=True,
            init='sobol',
            updating='deferred'
        )
        
        if result.success or (result.fun < best_score):
            candidate = result.x.reshape(N, 2)
            cand_score = _score(candidate)
            if cand_score < best_score and np.isfinite(cand_score):
                best_points = candidate
                best_score = cand_score
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 30:
        return best_points
    
    # Local polish with L-BFGS-B
    try:
        result2 = minimize(
            objective,
            best_points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result2.fun < best_score and np.isfinite(result2.fun):
            candidate = result2.x.reshape(N, 2)
            cand_score = _score(candidate)
            if cand_score < best_score:
                best_points = candidate
                best_score = cand_score
    except Exception:
        pass
    
    # Final check: ensure distinct points
    # If any two points are too close, perturb slightly
    for _ in range(100):
        diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        mask = ~np.eye(N, dtype=bool)
        min_d = np.min(dists[mask])
        if min_d > 1e-6:
            break
        # Perturb points that are too close
        for i in range(N):
            for j in range(i + 1, N):
                if dists[i, j] < 1e-6:
                    best_points[j] += np.random.uniform(-0.01, 0.01, size=2)
    
    return best_points
# EVOLVE-BLOCK-END
