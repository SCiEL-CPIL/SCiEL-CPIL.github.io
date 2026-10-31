"""Heilbronn convex regions - maximize min triangle area / convex hull area for 13 points."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# Precompute all C(13,3) = 286 triples
_TRIPLES = np.array(list(combinations(range(N), 3)), dtype=np.int32)

def _min_triangle_area(pts):
    """Compute minimum triangle area over all triples."""
    i = _TRIPLES[:, 0]
    j = _TRIPLES[:, 1]
    k = _TRIPLES[:, 2]
    v1 = pts[j] - pts[i]
    v2 = pts[k] - pts[i]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.min(np.abs(cross))

def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume = area
    except Exception:
        return 1e-10

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return -1e10
    return _min_triangle_area(pts) / hull_area

def _params_to_points(params):
    """
    Convert parameter vector to point coordinates.
    params: [a, b, theta, d0, d1, ..., d12]
    a, b: semi-axes of ellipse
    theta: rotation angle
    d0..d12: angular deviations from regular 13-gon
    """
    a = params[0]
    b = params[1]
    theta = params[2]
    deltas = params[3:]  # 13 angular deviations
    
    # Base angles for regular 13-gon
    base_angles = 2.0 * np.pi * np.arange(N) / N
    
    # Apply angular deviations
    angles = base_angles + deltas
    
    # Sort angles to maintain convex ordering
    # We need to ensure points form a convex polygon in order
    # Rotate so minimum angle is first, then sort
    angles = np.sort(angles)
    
    # Points on ellipse
    x = a * np.cos(angles)
    y = b * np.sin(angles)
    
    # Rotate by theta
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    xr = x * cos_t - y * sin_t
    yr = x * sin_t + y * cos_t
    
    return np.column_stack([xr, yr])

def _neg_score(params):
    """Negative score for minimization."""
    try:
        pts = _params_to_points(params)
        s = _score(pts)
        return -s
    except Exception:
        return 1e10

def construct():
    """Construct 13 points to maximize min triangle area / convex hull area."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial best: regular 13-gon on unit circle
    base_angles = 2.0 * np.pi * np.arange(N) / N
    best_pts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    best_score = _score(best_pts)
    
    # Stage 1: Differential Evolution for global optimization
    # Bounds: a in [0.3, 2.0], b in [0.3, 2.0], theta in [-pi, pi], deltas in [-0.5, 0.5]
    bounds = [(0.3, 2.0), (0.3, 2.0), (-np.pi, np.pi)] + [(-0.5, 0.5)] * N
    
    elapsed = time.time() - start_time
    if elapsed < 60.0:  # Only run DE if we have time
        # Determine number of iterations based on available time
        remaining = deadline - time.time()
        # Estimate: DE with popsize=20, 13+2=15 vars, each eval ~0.5ms
        # Budget ~80% of remaining time for DE
        de_time_budget = remaining * 0.7
        # Rough estimate of evals per second
        # Popsize 20, each generation = 20 evals, each ~1ms => ~20000 evals/sec
        # But let's be conservative
        max_iter = int(min(200, max(20, de_time_budget * 500)))  # conservative
        
        try:
            result = differential_evolution(
                _neg_score,
                bounds=bounds,
                seed=42,
                maxiter=max_iter,
                popsize=15,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=True,
                init='sobol',
                seed=42,
            )
            elapsed = time.time() - start_time
            if result.fun < -best_score:
                best_params = result.x
                best_pts = _params_to_points(best_params)
                best_score = _score(best_pts)
        except Exception:
            pass
    
    # Stage 2: Local polish with Nelder-Mead or BFGS
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        # Use the best params from DE or default to regular polygon params
        if 'best_params' in dir():
            x0 = best_params
        else:
            x0 = np.array([1.0, 1.0, 0.0] + [0.0] * N)
        
        try:
            # Nelder-Mead for non-smooth objective
            result2 = minimize(
                _neg_score,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12},
            )
            if result2.fun < -best_score:
                best_pts = _params_to_points(result2.x)
                best_score = _score(best_pts)
        except Exception:
            pass
        
        # Try BFGS too if time permits
        elapsed = time.time() - start_time
        if elapsed < deadline - 3.0:
            try:
                if 'best_params' in dir():
                    x0b = best_params
                else:
                    x0b = np.array([1.0, 1.0, 0.0] + [0.0] * N)
                result3 = minimize(
                    _neg_score,
                    x0b,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12},
                )
                if result3.fun < -best_score:
                    best_pts = _params_to_points(result3.x)
                    best_score = _score(best_pts)
            except Exception:
                pass
    
    # Final check: ensure we return before deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return whatever we have
        pass
    
    # Ensure points are well-formed
    if len(best_pts) != N or np.any(np.isnan(best_pts)):
        # Fallback to regular 13-gon
        base_angles = 2.0 * np.pi * np.arange(N) / N
        best_pts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    return best_pts
