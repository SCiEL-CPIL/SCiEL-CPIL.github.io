"""
Strategy: Global optimization using Differential Evolution.
The previous approach used local refinement from the regular octagon.
This approach uses a population-based global search to escape local minima
and find configurations that are not "near-regular" octagons.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

def _score(points):
    """
    Compute Delta / n^n for 8 points.
    Delta is the product of ordered pair distances after scaling to diameter 2.
    """
    n = len(points)
    pts = np.array(points, dtype=float)
    
    # Ensure points are 2D
    if pts.shape[1] != 2:
        return 0.0
        
    # Compute all pairwise distances
    # Using broadcasting
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Diameter = max distance
    diam = np.max(dists)
    if diam < 1e-15:
        return 0.0
    
    # Scale to diameter 2
    # We only care about the product of non-zero distances
    # dists[i,i] is 0, so we exclude diagonal
    mask = ~np.eye(n, dtype=bool)
    ordered_dists = dists[mask]
    
    # Scale distances
    scaled_dists = ordered_dists / diam * 2.0
    
    # Check for distinct points (distance > 0)
    if np.any(scaled_dists < 1e-12):
        return 0.0
        
    # Product of ordered pair distances
    # log(Delta) = sum(log(scaled_dists))
    log_delta = np.sum(np.log(scaled_dists))
    
    # Score = Delta / n^n
    # log(Score) = log_delta - n*log(n)
    log_score = log_delta - n * np.log(n)
    
    # Avoid overflow/underflow in exp if possible, but score is usually O(1)
    return np.exp(log_score)

def _neg_score(params):
    """
    Objective function for optimization.
    params: 16 floats (x0, y0, ..., x7, y7)
    """
    pts = params.reshape(8, 2)
    return -_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Bounds for DE: 
    # Since we normalize to diameter 2, the absolute scale doesn't matter for the score,
    # but DE needs a bounded box. 
    # A box of [-10, 10] is large enough to contain any interesting geometry 
    # after normalization, and prevents numerical issues with very large/small coords.
    bounds = [(-10.0, 10.0)] * 16
    
    # Initial population size
    popsize = 50
    
    try:
        # Use Differential Evolution for global search
        # tol and maxiter control convergence
        # We give it a generous budget but monitor time
        res = differential_evolution(
            _neg_score, 
            bounds, 
            seed=42,
            maxiter=1000,
            popsize=popsize,
            tol=1e-10,
            mutation=(0.5, 1.5),
            recombination=0.9,
            polish=False,  # DE's internal polish might be slow or redundant if we do our own
            updating='deferred',
            workers=1
        )
        
        best_params = res.x
        best_score = -res.fun
        
    except Exception as e:
        # Fallback if DE fails or times out unexpectedly
        # Try a few random starts with Nelder-Mead as a backup
        best_params = np.zeros(16)
        best_score = -_neg_score(best_params)
        
        rng = np.random.RandomState(42)
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            start = rng.uniform(-10, 10, 16)
            try:
                from scipy.optimize import minimize
                res_nm = minimize(_neg_score, start, method='Nelder-Mead', options={'maxiter': 5000})
                if -res_nm.fun > best_score:
                    best_score = -res_nm.fun
                    best_params = res_nm.x
            except:
                pass

    # Refine the best result found by DE using a local method (Nelder-Mead)
    # DE might stop near the optimum but not exactly on it.
    if time.time() < deadline - 10.0:
        try:
            from scipy.optimize import minimize
            # Use the best point from DE as start for local refinement
            res_local = minimize(
                _neg_score, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if -res_local.fun > best_score:
                best_score = -res_local.fun
                best_params = res_local.x
        except Exception:
            pass

    # Convert params to points
    points = best_params.reshape(8, 2)
    
    # Final sanity check
    # Ensure distinctness
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-8:
        # Fallback to regular octagon if points collapsed
        octagon = np.array([[np.cos(2 * np.pi * k / 8), np.sin(2 * np.pi * k / 8)] for k in range(8)])
        points = octagon
        
    # Ensure coordinates are within bounds (they should be, but just in case)
    # The problem requires |coord| < 1e6. Our search was in [-10, 10], so this is safe.
    
    return [[float(p[0]), float(p[1])] for p in points]
# EVOLVE-BLOCK-END
