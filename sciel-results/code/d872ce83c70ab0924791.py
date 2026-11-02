"""Heilbronn convex regions: maximize min triangle area over convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

_DEADLINE = 150.0
_START = time.time()
_SAFETY = 15.0  # seconds before deadline to stop

# Precompute all triangle index triplets for vectorized area calculation
_TRI_INDICES = np.array(list(combinations(range(N), 3)))
_NUM_TRI = len(_TRI_INDICES)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _min_triangle_area(pts):
    """Minimum area of any triangle formed by 3 of the points (vectorized)."""
    # pts: (n, 2)
    # v1 = pts[j] - pts[i]
    # v2 = pts[k] - pts[i]
    # area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
    
    # Vectorized calculation
    # We can compute cross product for all triangles at once
    # v1 = pts[_J] - pts[_I]
    # v2 = pts[_K] - pts[_I]
    # cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    # areas = 0.5 * abs(cross)
    
    v1 = pts[_J] - pts[_I]
    v2 = pts[_K] - pts[_I]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return np.min(np.abs(cross)) * 0.5

def _convex_hull_area(pts):
    """Area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # in 2D, volume == area
    except Exception:
        # Fallback: shoelace on sorted points
        order = np.lexsort((pts[:, 1], pts[:, 0]))
        p = pts[order]
        x, y = p[:, 0], p[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _score(pts):
    """min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    return _min_triangle_area(pts) / hull_area

def _neg_score(x):
    """Negative score for minimization. x is flattened array of shape (2*N,)."""
    pts = x.reshape(N, 2)
    # Basic validity check: if points are too close or degenerate, penalize
    if np.any(np.isnan(pts)):
        return 1e10
    s = _score(pts)
    return -s

def construct():
    global _START, _DEADLINE
    elapsed = time.time() - _START
    remaining = _DEADLINE - _SAFETY - elapsed
    
    if remaining < 10.0:
        # Fallback: return a reasonable heuristic immediately
        return _regular_polygon_init(N)

    # Define bounds for the search space. 
    # We can normalize the convex hull area to 1 eventually, but during optimization,
    # we work in a fixed box to avoid scale issues. 
    # The objective is scale-invariant (ratio), so the absolute scale doesn't matter,
    # but the optimizer needs a defined space.
    # Let's search in a box [0, 1] x [0, 1].
    bounds = [(0, 1), (0, 1)] * N
    
    # Strategy:
    # 1. Run Differential Evolution for a significant portion of the time.
    # 2. Take the best result and refine it with Nelder-Mead or L-BFGS-B.
    
    best_pts = _regular_polygon_init(N)
    best_score = _score(best_pts)
    
    # Time budget allocation
    # DE is slow per iteration but robust. 
    # Let's give it ~60-80% of the time.
    de_time_budget = min(remaining * 0.7, 80.0) # Cap at 80s for DE to leave time for refinement
    de_max_iter = 100 # DE iterations can be expensive, but we will use a callback to stop early
    
    def _de_callback(xk, convergence):
        if time.time() - _START > _DEADLINE - _SAFETY - 20:
            return True
        return False

    try:
        # Use DE with a moderate population size
        # 'maxiter' is the number of generations. 
        # We will rely on the callback to stop if time runs out.
        result_de = differential_evolution(
            _neg_score, 
            bounds, 
            maxiter=50, 
            popsize=15,
            tol=1e-4,
            seed=42,
            callback=_de_callback,
            polish=False # We will do our own polish
        )
        
        if result_de.success or result_de.fun < -best_score:
            candidate = result_de.x.reshape(N, 2)
            s = _score(candidate)
            if s > best_score:
                best_score = s
                best_pts = candidate.copy()
    except Exception as e:
        pass

    # Refinement phase
    # Use the best point found so far (either from DE or initial heuristic)
    # Nelder-Mead is good for local polishing of non-smooth functions (min of areas is not smooth)
    
    remaining_for_refine = _DEADLINE - _SAFETY - (time.time() - _START)
    if remaining_for_refine > 5.0:
        x0 = best_pts.flatten()
        
        def _nm_callback(xk):
            if time.time() - _START > _DEADLINE - _SAFETY:
                return True
            return False

        try:
            result_nm = minimize(
                _neg_score, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True},
                callback=_nm_callback
            )
            
            if result_nm.fun < -best_score:
                candidate = result_nm.x.reshape(N, 2)
                s = _score(candidate)
                if s > best_score:
                    best_score = s
                    best_pts = candidate.copy()
        except Exception:
            pass

    return best_pts

def _regular_polygon_init(n, center=(0.5, 0.5), radius=0.4):
    """Place n points: regular (n-1)-gon around center + 1 center point."""
    m = n - 1
    angles = 2 * np.pi * np.arange(m) / m
    pts = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])
    pts = np.vstack([pts, np.array(center)])
    return pts
# EVOLVE-BLOCK-END
