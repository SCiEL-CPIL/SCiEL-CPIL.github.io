"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

N = 14

# EVOLVE-BLOCK-START

def _triangle_areas(pts):
    """Compute areas of all C(N,3) triangles from N points."""
    n = len(pts)
    areas = np.empty((n * (n - 1) * (n - 2)) // 6)
    idx = 0
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(pj - pi, pk - pi)|
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                areas[idx] = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                idx += 1
    return areas

def _score(pts):
    """Minimum triangle area divided by convex hull area."""
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area < 1e-12:
            return 0.0
        areas = _triangle_areas(pts)
        min_area = np.min(areas)
        return min_area / hull_area
    except Exception:
        return 0.0

def _regular_polygon(n, radius=1.0):
    """Generate vertices of a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

def _two_ring_config(n_outer=7, n_inner=7, r_inner=0.5):
    """Generate a two-ring configuration: outer regular n_outer-gon and inner regular n_inner-gon (rotated)."""
    outer = _regular_polygon(n_outer, radius=1.0)
    # Rotate inner ring by half the angular step to interleave
    inner_angles = np.linspace(np.pi / n_inner, 2 * np.pi + np.pi / n_inner, n_inner, endpoint=False)
    inner = np.column_stack((r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)))
    return np.vstack([outer, inner])

def _optimize_from_init(init_pts, time_budget=10.0):
    """Local optimization from an initial configuration using Nelder-Mead."""
    best_pts = init_pts.copy()
    best_score = _score(best_pts)
    start = time.time()
    
    def objective(x):
        pts = x.reshape(N, 2)
        return -_score(pts)
    
    x0 = best_pts.flatten()
    
    while time.time() - start < time_budget:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        new_pts = res.x.reshape(N, 2)
        new_score = -res.fun
        if new_score > best_score:
            best_pts = new_pts
            best_score = new_score
        # Check if we've converged
        if res.success and res.nit < 10:
            break
        # Perturb slightly to escape local minima
        rng = np.random.default_rng()
        x0 = best_pts.flatten() + rng.normal(0, 0.01, size=2 * N)
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with multiple diverse initial configurations
    init_configs = []
    
    # 1. Regular 14-gon
    init_configs.append(_regular_polygon(14, 1.0))
    
    # 2. Two-ring: 7 outer + 7 inner
    for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7]:
        init_configs.append(_two_ring_config(7, 7, r_inner))
    
    # 3. Two-ring: 5 outer + 9 inner
    for r_inner in [0.4, 0.5, 0.6]:
        outer = _regular_polygon(5, 1.0)
        inner_angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
        inner = np.column_stack((r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)))
        init_configs.append(np.vstack([outer, inner]))
    
    # 4. Two-ring: 9 outer + 5 inner
    for r_inner in [0.4, 0.5, 0.6]:
        outer = _regular_polygon(9, 1.0)
        inner_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        inner = np.column_stack((r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)))
        init_configs.append(np.vstack([outer, inner]))
    
    # 5. Random points in unit disk
    rng = np.random.default_rng(42)
    for _ in range(5):
        angles = rng.uniform(0, 2 * np.pi, N)
        radii = rng.uniform(0.3, 1.0, N)
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        init_configs.append(pts)
    
    # Evaluate all initial configs
    best_pts = None
    best_score = -np.inf
    for cfg in init_configs:
        if time.time() > deadline - 20.0:
            break
        s = _score(cfg)
        if s > best_score:
            best_score = s
            best_pts = cfg.copy()
    
    if best_pts is None:
        return _regular_polygon(14, 1.0)
    
    # Local optimization from the best initial config
    if time.time() < deadline - 15.0:
        optimized_pts, opt_score = _optimize_from_init(best_pts, time_budget=deadline - time.time() - 15.0)
        if opt_score > best_score:
            best_pts = optimized_pts
            best_score = opt_score
    
    # Try optimizing from a few other good initial configs
    if time.time() < deadline - 10.0:
        for cfg in init_configs[:5]:
            if time.time() > deadline - 10.0:
                break
            if np.allclose(cfg, best_pts, atol=1e-6):
                continue
            s = _score(cfg)
            if s >= best_score * 0.9:  # Only optimize promising starts
                opt_pts, opt_s = _optimize_from_init(cfg, time_budget=min(10.0, deadline - time.time() - 10.0))
                if opt_s > best_score:
                    best_pts = opt_pts
                    best_score = opt_s
    
    return best_pts

# EVOLVE-BLOCK-END
