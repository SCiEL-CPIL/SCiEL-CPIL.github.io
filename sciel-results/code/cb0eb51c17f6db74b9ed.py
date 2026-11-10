# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize

# Deadline guard: must finish within 600s, leave safety margin
_DEADLINE = 600.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - 5.0 - (time.time() - _START_TIME)

def _metric(points):
    """Compute (d_min / d_max)^2. Return 0 if invalid."""
    try:
        if not np.all(np.isfinite(points)):
            return 0.0
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0 or d_max < 1e-15:
            return 0.0
        if d_min < 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio
    except Exception:
        return 0.0

def _neg_metric_flat(x):
    """Objective for optimizers: minimize -metric."""
    points = x.reshape(-1, 3)
    return -_metric(points)

def _refine_local(points, max_iter=200):
    """Nelder-Mead refinement."""
    try:
        res = minimize(
            _neg_metric_flat,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        refined = res.x.reshape(-1, 3)
        if _metric(refined) > _metric(points):
            return refined
    except Exception:
        pass
    return points

def _run_de(bounds, popsize=20, maxiter=100, seed=42):
    """Run differential evolution."""
    try:
        res = differential_evolution(
            _neg_metric_flat,
            bounds=bounds,
            popsize=popsize,
            maxiter=maxiter,
            seed=seed,
            tol=1e-8,
            polish=False,
            init='latinhypercube'
        )
        return res.x.reshape(-1, 3)
    except Exception:
        return None

def _symmetrize_and_center(points):
    """Center points at origin and optionally scale."""
    points = points - np.mean(points, axis=0)
    return points

def _try_multiple_seeds(bounds, n_seeds=5, de_popsize=20, de_maxiter=80, nm_maxiter=150):
    """Try multiple random initializations via DE + NM refinement."""
    best_points = None
    best_score = 0.0
    
    for seed in range(n_seeds):
        if _time_left() < 30:
            break
        
        points = _run_de(bounds, popsize=de_popsize, maxiter=de_maxiter, seed=seed)
        if points is None:
            continue
        
        points = _symmetrize_and_center(points)
        points = _refine_local(points, max_iter=nm_maxiter)
        
        score = _metric(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    return best_points, best_score

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    
    Returns
        points: np.ndarray of shape (14,3)
    """
    n = 14
    d = 3
    total_params = n * d
    
    # Bounds: allow points in [-5, 5]^3
    bounds = [(-5.0, 5.0)] * total_params
    
    best_points = None
    best_score = 0.0
    
    # Strategy 1: Multiple DE + NM runs with different seeds
    # Use moderate settings to stay within time budget
    try:
        points, score = _try_multiple_seeds(
            bounds, 
            n_seeds=5, 
            de_popsize=25, 
            de_maxiter=100, 
            nm_maxiter=200
        )
        if points is not None and score > best_score:
            best_score = score
            best_points = points
    except Exception:
        pass
    
    # Strategy 2: If time remains, try more seeds or larger popsize
    if _time_left() > 60 and best_points is not None:
        try:
            points, score = _try_multiple_seeds(
                bounds,
                n_seeds=3,
                de_popsize=35,
                de_maxiter=150,
                nm_maxiter=300
            )
            if points is not None and score > best_score:
                best_score = score
                best_points = points
        except Exception:
            pass
    
    # Strategy 3: Final local refinement on the best found
    if best_points is not None and _time_left() > 10:
        try:
            refined = _refine_local(best_points, max_iter=500)
            if _metric(refined) > best_score:
                best_points = refined
                best_score = _metric(refined)
        except Exception:
            pass
    
    # Fallback: if no valid points found, return a reasonable default
    if best_points is None:
        np.random.seed(42)
        best_points = np.random.randn(n, d)
        best_points = best_points - np.mean(best_points, axis=0)
    
    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        np.random.seed(42)
        best_points = np.random.randn(n, d)
    
    return best_points

# EVOLVE-BLOCK-END
