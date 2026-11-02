# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4
DEADLINE = 140

_TRIPLETS = list(combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

def _project_to_triangle(pts):
    y = np.clip(pts[:, 1], 0.0, S3H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(pts[:, 0], x_min, x_max)
    return np.column_stack([x, y])

def _min_triangle_area(pts):
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)

def _repulsion(pts):
    # Sum of 1/d^2 for all pairs
    # Vectorized: compute all pairwise distances
    # d^2_{ij} = (xi-xj)^2 + (yi-yj)^2
    # We want sum_{i<j} 1/d^2_{ij}
    # Using broadcasting
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    # Mask diagonal
    dist_sq[np.arange(N), np.arange(N)] = 1.0 # Avoid division by zero
    # To avoid double counting and self-interaction, we can sum upper triangle
    # Or just sum all and divide by 2, but mask diagonal
    inv_dist_sq = 1.0 / (dist_sq + 1e-10)
    # Zero out diagonal
    inv_dist_sq[np.arange(N), np.arange(N)] = 0.0
    return 0.5 * np.sum(inv_dist_sq)

def _objective(pts_flat, lam):
    pts = _project_to_triangle(pts_flat.reshape(-1, 2))
    min_area = _min_triangle_area(pts)
    rep = _repulsion(pts)
    return -min_area + lam * rep

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts.copy())
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # Structured initial points
    # Vertices
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, S3H])
    
    # Midpoints
    m01 = (v0 + v1) / 2
    m12 = (v1 + v2) / 2
    m20 = (v2 + v0) / 2
    
    # Centroid
    c = (v0 + v1 + v2) / 3
    
    # A known good configuration for n=11?
    # Let's use a grid-like structure or just random perturbations of the above.
    base_points = [v0, v1, v2, m01, m12, m20, c]
    
    # Add 4 more points to make 11
    # Place them in a small cluster around centroid or spread out
    # Let's try placing them at 1/3 and 2/3 along medians
    # Or just randomize the last 4
    init_candidates = []
    
    # Candidate 1: Symmetric
    p1 = np.array([
        v0, v1, v2,
        m01, m12, m20,
        c,
        (2*v0 + v1 + v2)/4, # barycentric (2,1,1)/4
        (v0 + 2*v1 + v2)/4,
        (v0 + v1 + 2*v2)/4,
        (v0 + v1 + v2)/3 + np.array([0.1, 0.1]) # Perturbed centroid
    ])
    init_candidates.append(p1)
    
    # Candidate 2: Random
    for _ in range(5):
        u = rng.random((N, 2))
        # Map to triangle
        # Use barycentric coords
        b = rng.random(N)
        c = rng.random(N)
        mask = b + c > 1
        b[mask] = 1 - b[mask]
        c[mask] = 1 - c[mask]
        x = b + 0.5 * c
        y = c * S3H
        p = np.column_stack([x, y])
        init_candidates.append(p)

    # Annealing schedule for lambda (repulsion strength)
    # Start high to spread points, end low to optimize area
    lambda_schedule = [50.0, 20.0, 5.0, 1.0, 0.1, 0.0]
    
    max_restarts = 10
    restarts_done = 0
    
    for init_idx in range(len(init_candidates)):
        if not _elapsed_ok(10):
            break
            
        pts = _project_to_triangle(init_candidates[init_idx].copy())
        _update_best(pts)
        
        # Run annealing Nelder-Mead
        current_pts = pts.copy()
        
        for lam in lambda_schedule:
            if not _elapsed_ok(5):
                break
                
            def obj_func(x, l=lam):
                return _objective(x, l)
            
            try:
                res = minimize(obj_func, current_pts.flatten(), method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
                current_pts = _project_to_triangle(res.x.reshape(-1, 2))
                _update_best(current_pts)
            except Exception:
                pass
        
        restarts_done += 1
        # If we have time, try more random restarts
        if _elapsed_ok(20) and restarts_done < max_restarts:
             # Generate a new random candidate
             b = rng.random(N)
             c = rng.random(N)
             mask = b + c > 1
             b[mask] = 1 - b[mask]
             c[mask] = 1 - c[mask]
             x = b + 0.5 * c
             y = c * S3H
             p = np.column_stack([x, y])
             init_candidates.append(p)
             restarts_done += 1 # Count as used slot

    # Final polish with exact min area (lambda=0) using Nelder-Mead
    if best_pts is not None and _elapsed_ok(5):
        def final_obj(x):
            return _objective(x, 0.0)
        
        try:
            res = minimize(final_obj, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            final_pts = _project_to_triangle(res.x.reshape(-1, 2))
            _update_best(final_pts)
        except Exception:
            pass

    if best_pts is None:
        pts = np.array([rng.random(2) for _ in range(N)])
        return _project_to_triangle(pts)

    return _project_to_triangle(best_pts)
# EVOLVE-BLOCK-END
