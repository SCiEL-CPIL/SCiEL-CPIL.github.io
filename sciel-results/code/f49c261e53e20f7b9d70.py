"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _score(points):
    """Compute Delta / n^n for 8 points, where Delta is product of ordered pair distances
    after scaling to diameter 2."""
    n = len(points)
    pts = np.array(points, dtype=float)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Diameter = max distance
    diam = np.max(dists)
    if diam < 1e-15:
        return 0.0
    # Scale to diameter 2
    scaled_dists = dists / diam * 2.0
    # Product of ordered pair distances (exclude self-distances which are 0)
    # We need product over all ordered pairs (i,j) with i != j
    mask = ~np.eye(n, dtype=bool)
    ordered_dists = scaled_dists[mask]
    # Delta = product of all ordered pair distances
    # Use log to avoid underflow
    log_delta = np.sum(np.log(ordered_dists))
    log_score = log_delta - n * np.log(n)
    return np.exp(log_score)

def _neg_score(points):
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Start with regular octagon
    from math import cos, sin, pi
    octagon = [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
    best_points = np.array(octagon, dtype=float)
    best_score = _score(best_points)
    
    # Try to improve using local optimization
    # Use Nelder-Mead or L-BFGS-B on the 16 coordinates
    try:
        # First try with a slightly perturbed regular octagon
        rng = np.random.RandomState(42)
        
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            
            if restart == 0:
                x0 = best_points.flatten()
            else:
                # Perturb the regular octagon
                angles = 2 * pi * np.arange(8) / 8
                radii = 1.0 + 0.1 * rng.randn(8)
                angles = angles + 0.05 * rng.randn(8)
                x0 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).flatten()
            
            result = minimize(_neg_score, x0, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if time.time() > deadline - 5.0:
                break
            
            candidate = result.x.reshape(8, 2)
            # Check distinctness
            diff = candidate[:, np.newaxis, :] - candidate[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) < 1e-8:
                continue
            
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate
                
    except Exception:
        pass
    
    # Return the best points found
    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
