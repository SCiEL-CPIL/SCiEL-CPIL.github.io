"""Rewritten program for Heilbronn convex regions n=14."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

def _hull_area(p):
    """Return area of the convex hull of point set p."""
    try:
        hull = ConvexHull(p)
        return hull.volume
    except Exception:
        return 1e-10

def _all_triangle_areas(p):
    """
    Return array of all triangle areas (true area, not 2*area) for all triples.
    Shape: (C(n,3),)
    """
    n = len(p)
    indices = np.array(list(itertools.combinations(range(n), 3)))
    # Vectorized calculation
    # p[i], p[j], p[k]
    pi = p[indices[:, 0]]
    pj = p[indices[:, 1]]
    pk = p[indices[:, 2]]
    
    # Area = 0.5 * |det(pj-pi, pk-pi)|
    # det = (pj_x - pi_x)*(pk_y - pi_y) - (pj_y - pi_y)*(pk_x - pi_x)
    det = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
          (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(det)
    return areas

def _min_triangle_area(p):
    """Return minimum triangle area over all triples."""
    areas = _all_triangle_areas(p)
    if len(areas) == 0:
        return 0.0
    return np.min(areas)

def _score(p):
    """Return min_triangle_area / convex_hull_area."""
    hull_area = _hull_area(p)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(p)
    return min_area / hull_area

def _objective(x):
    """
    Objective function for minimization.
    x: flat array of 2*N coordinates.
    Returns negative score (since we want to maximize score).
    """
    p = x.reshape(-1, 2)
    s = _score(p)
    return -s

def _initial_regular_polygon():
    """Generate points of a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack([x, y])

def _initial_two_rings():
    """Generate points in two concentric rings."""
    # 7 points on outer ring, 7 points on inner ring
    # Offset inner ring to maximize min distance
    n_outer = 7
    n_inner = 7
    r_outer = 1.0
    r_inner = 0.5
    
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    
    x_outer = r_outer * np.cos(angles_outer)
    y_outer = r_outer * np.sin(angles_outer)
    
    x_inner = r_inner * np.cos(angles_inner)
    y_inner = r_inner * np.sin(angles_inner)
    
    x = np.concatenate([x_outer, x_inner])
    y = np.concatenate([y_outer, y_inner])
    return np.column_stack([x, y])

def _optimize_full_points(initial_pts, start_time, deadline):
    """
    Optimize the full set of 2*N coordinates using L-BFGS-B.
    """
    x0 = initial_pts.flatten()
    
    # Bounds: keep points within a reasonable box, say [-1.5, 1.5]
    # This prevents points from flying off to infinity which can degenerate the hull
    bounds = [(-1.5, 1.5)] * (2 * N)
    
    best_x = x0
    best_score = _score(initial_pts)
    
    # Use L-BFGS-B which is good for this kind of smooth-ish (but non-smooth min) problem
    # We might need multiple restarts or different methods if it gets stuck
    try:
        res = minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.fun < -best_score:
            candidate_pts = res.x.reshape(-1, 2)
            candidate_score = _score(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_x = res.x
    except Exception:
        pass
    
    # Try Powell as well, which is derivative-free and robust to non-smoothness
    if time.time() < deadline - 10:
        try:
            res_p = minimize(
                _objective,
                best_x,
                method='Powell',
                bounds=bounds,
                options={'maxiter': 200, 'xtol': 1e-6, 'ftol': 1e-6}
            )
            if res_p.fun < -best_score:
                candidate_pts = res_p.x.reshape(-1, 2)
                candidate_score = _score(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_x = res_p.x
        except Exception:
            pass
            
    return best_x.reshape(-1, 2), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_pts = None
    best_score = 0.0
    
    # Strategy 1: Start from regular polygon
    init1 = _initial_regular_polygon()
    pts1, score1 = _optimize_full_points(init1, start_time, deadline)
    if score1 > best_score:
        best_score = score1
        best_pts = pts1
        
    # Strategy 2: Start from two rings
    if time.time() < deadline - 20:
        init2 = _initial_two_rings()
        pts2, score2 = _optimize_full_points(init2, start_time, deadline)
        if score2 > best_score:
            best_score = score2
            best_pts = pts2
            
    # Strategy 3: Random restarts with local optimization
    if time.time() < deadline - 10:
        rng = np.random.default_rng(42)
        for i in range(5):
            if time.time() > deadline - 5:
                break
            
            # Generate random points in a circle
            angles = rng.uniform(0, 2 * np.pi, N)
            radii = rng.uniform(0.5, 1.0, N)
            x = radii * np.cos(angles)
            y = radii * np.sin(angles)
            init_rand = np.column_stack([x, y])
            
            # Quick local optimization
            pts_r, score_r = _optimize_full_points(init_rand, start_time, deadline)
            if score_r > best_score:
                best_score = score_r
                best_pts = pts_r
                
    # Final polish on the best result found
    if best_pts is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(
                _objective,
                best_pts.flatten(),
                method='L-BFGS-B',
                bounds=[(-1.5, 1.5)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res_final.fun < -best_score:
                candidate_pts = res_final.x.reshape(-1, 2)
                candidate_score = _score(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pts = candidate_pts
        except Exception:
            pass

    # Normalize to unit area? The problem says "normalized to unit area" but the score
    # is "minimum triangle area divided by the convex hull area".
    # This ratio is scale-invariant. So we don't need to explicitly scale to unit area
    # for the score calculation, but the evaluator might check if the hull area is 1?
    # "The notebook normalizes by the point set convex hull area."
    # "score is the minimum triangle area divided by the convex hull area"
    # So the raw points can be any scale. The ratio handles it.
    # However, if the evaluator expects the points to be in a specific normalized space,
    # it might be safer to return them as is, since the score function provided in the
    # contract uses the ratio.
    
    if best_pts is None:
        return _initial_regular_polygon()
        
    return best_pts
# EVOLVE-BLOCK-END
