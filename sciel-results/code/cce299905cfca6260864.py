"""Heilbronn triangles: maximize the smallest triangle area for 11 points in an equilateral triangle."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
_H = np.sqrt(3) / 2.0
_SQRT3 = np.sqrt(3.0)

# Precompute all C(11,3)=165 triangle index triples
_combos = list(combinations(range(N), 3))
_ci = np.array([c[0] for c in _combos])
_cj = np.array([c[1] for c in _combos])
_ck = np.array([c[2] for c in _combos])


def _min_area(pts):
    """Minimum area of any triangle formed by 3 of the 11 points."""
    pi = pts[_ci]
    pj = pts[_cj]
    pk = pts[_ck]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return float(np.min(areas))


def _inside_triangle(pts, tol=1e-9):
    """Check all points are inside (or on boundary of) the equilateral triangle."""
    x = pts[:, 0]
    y = pts[:, 1]
    if np.any(y < -tol):
        return False
    if np.any(y > _H * x + tol):
        return False
    if np.any(y > _H * (1.0 - x) + tol):
        return False
    if np.any(x < -tol) or np.any(x > 1.0 + tol):
        return False
    return True


def _project(pts):
    """Project points onto the triangle (clamp to feasible region)."""
    pts = pts.copy()
    x = pts[:, 0]
    y = pts[:, 1]
    # x in [0, 1]
    x = np.clip(x, 0.0, 1.0)
    # y >= 0
    y = np.maximum(y, 0.0)
    # y <= min(_H*x, _H*(1-x))
    upper = np.minimum(_H * x, _H * (1.0 - x))
    y = np.minimum(y, upper)
    pts[:, 0] = x
    pts[:, 1] = y
    return pts


def _neg_min_area_flat(flat_pts):
    """Objective for minimization: negative of minimum triangle area."""
    pts = flat_pts.reshape(N, 2)
    pts = _project(pts)
    return -_min_area(pts)


def _initial_config_symmetric():
    """
    Symmetric arrangement based on literature/heuristics for 11 points.
    This config places points at vertices and midpoints of sub-triangles,
    providing a good starting point for local search.
    """
    h = _H
    pts = np.array([
        [0.0, 0.0],       # 0: BL
        [1.0, 0.0],       # 1: BR
        [0.5, h],         # 2: Top
        [0.5, 0.0],       # 3: Bottom Mid
        [0.75, h/2.0],    # 4: Right Mid
        [0.25, h/2.0],    # 5: Left Mid
        [0.5, h/3.0],     # 6: Center
        [1/6.0, h/6.0],   # 7: BL Centroid
        [5/6.0, h/6.0],   # 8: BR Centroid
        [0.5, 2*h/3.0],   # 9: Top Centroid
        [0.625, h/4.0]    # 10: Extra
    ])
    return _project(pts)


def _max_min_dist_surrogate(flat_pts):
    """
    Surrogate objective: maximize the minimum distance between any pair of points.
    This is a smoother, easier-to-optimize problem that often leads to good
    configurations for the min-triangle-area problem.
    """
    pts = flat_pts.reshape(N, 2)
    pts = _project(pts)
    
    # Compute all pairwise distances
    # Shape (N, N)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Exclude diagonal (self-distance)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return -min_dist


def _local_optimize_nelder_mead(pts, max_time=20.0):
    """Run Nelder-Mead optimization starting from pts."""
    t0 = time.time()
    best_pts = pts.copy()
    best_score = _min_area(pts)
    
    flat_pts = pts.flatten()
    
    try:
        # Nelder-Mead is derivative-free and handles non-smoothness better
        res = minimize(
            _neg_min_area_flat,
            flat_pts,
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        
        if time.time() - t0 < max_time:
            opt_pts = res.x.reshape(N, 2)
            opt_pts = _project(opt_pts)
            opt_score = _min_area(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
    except Exception:
        pass
    
    return best_pts, best_score


def _optimize_via_surrogate(pts, max_time=20.0):
    """
    Two-stage optimization:
    1. Optimize the max-min distance surrogate (smoother, easier).
    2. Refine with the true min-triangle-area objective.
    """
    t0 = time.time()
    best_pts = pts.copy()
    best_score = _min_area(pts)
    
    flat_pts = pts.flatten()
    
    # Stage 1: Optimize surrogate
    try:
        res1 = minimize(
            _max_min_dist_surrogate,
            flat_pts,
            method='Nelder-Mead',
            options={
                'maxiter': 300,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        pts1 = res1.x.reshape(N, 2)
        pts1 = _project(pts1)
        score1 = _min_area(pts1)
        if score1 > best_score:
            best_score = score1
            best_pts = pts1.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() - t0 > max_time * 0.6:
        return best_pts, best_score
    
    # Stage 2: Refine with true objective
    best_pts, best_score = _local_optimize_nelder_mead(best_pts, max_time=max_time - (time.time() - t0))
    
    return best_pts, best_score


def construct():
    t_start = time.time()
    # Safety margin: 150s limit, keep 10s buffer
    deadline = t_start + 140.0

    # 1. Start with the symmetric configuration
    best_pts = _initial_config_symmetric()
    best_score = _min_area(best_pts)

    # 2. Run surrogate-based optimization from this start
    if time.time() < deadline - 10:
        best_pts, best_score = _optimize_via_surrogate(best_pts, max_time=min(25.0, deadline - time.time() - 5))

    # 3. Multi-start: Perturb the best solution and re-optimize
    rng = np.random.RandomState(42)

    # Define perturbation scales
    scales = [0.05, 0.02, 0.01, 0.1]

    attempts = 0
    while time.time() < deadline - 5:
        # Choose a scale, cycle through them
        scale = scales[attempts % len(scales)]
        
        # Perturb the current best
        perturbed = best_pts + rng.randn(N, 2) * scale
        perturbed = _project(perturbed)
        
        # Ensure the perturbation is valid and distinct
        if not _inside_triangle(perturbed):
            attempts += 1
            continue
            
        # Run optimization from this perturbed start
        # Limit time per attempt to ensure we can do many
        time_left = deadline - time.time() - 2
        if time_left < 2:
            break
            
        # Adaptive time limit: shorter for later attempts
        # If we are getting close to deadline, do fewer iterations
        max_iter_time = min(10.0, time_left * 0.8)
        
        # Alternate between surrogate-based and direct optimization
        if attempts % 2 == 0:
            opt_pts, opt_score = _optimize_via_surrogate(perturbed, max_time=max_iter_time)
        else:
            opt_pts, opt_score = _local_optimize_nelder_mead(perturbed, max_time=max_iter_time)
        
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts.copy()
            
        attempts += 1
        
        # If we've done enough attempts and score is stable, we might stop early?
        # No, keep going until time is up to explore more.
        
    # Final validation
    best_pts = _project(best_pts)
    if not _inside_triangle(best_pts):
        # Fallback to initial config if something went wrong
        best_pts = _project(_initial_config_symmetric())
        
    return best_pts
