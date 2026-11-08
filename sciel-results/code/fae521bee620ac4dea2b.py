"""Heilbronn triangles: maximize min triangle area for 11 points in unit equilateral triangle."""
N = 11
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

# Precompute triangle indices
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])


def _min_triangle_area(points):
    """Compute minimum area of all C(11,3) triangles."""
    pi = points[_idx_i]
    pj = points[_idx_j]
    pk = points[_idx_k]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)


def _project_to_triangle(pts):
    """Project points to stay inside the equilateral triangle."""
    h = np.sqrt(3) / 2.0
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0.0
        if y > h * x:
            y = h * x
        if y > h * (1.0 - x):
            y = h * (1.0 - x)
        pts[i] = [x, y]
    return pts


def _objective_flat(flat_pts):
    """Negative min triangle area for minimization."""
    pts = flat_pts.reshape(N, 2)
    pts = _project_to_triangle(pts.copy())
    return -_min_triangle_area(pts)


def _make_config(pattern_id, rng=None, jitter=0.0):
    """Generate symmetric configurations."""
    h = np.sqrt(3) / 2.0
    
    configs = {
        0: np.array([
            [0.5, 0.0], [0.0, 0.0], [1.0, 0.0],
            [0.25, h*0.25], [0.75, h*0.25],
            [0.5, h*0.5],
            [0.125, h*0.125], [0.875, h*0.125],
            [0.375, h*0.375], [0.625, h*0.375],
            [0.5, h*0.875],
        ]),
        1: np.array([
            [0.0, 0.0], [0.25, 0.0], [0.5, 0.0], [0.75, 0.0], [1.0, 0.0],
            [0.125, h*0.25], [0.375, h*0.25], [0.625, h*0.25], [0.875, h*0.25],
            [0.25, h*0.5], [0.75, h*0.5],
        ]),
        2: np.array([
            [0.0, 0.0], [0.33, 0.0], [0.67, 0.0], [1.0, 0.0],
            [0.15, h*0.25], [0.45, h*0.25], [0.75, h*0.25], [0.95, h*0.25],
            [0.3, h*0.5], [0.5, h*0.6], [0.7, h*0.5],
        ]),
        3: np.array([
            [0.5, 0.0], [0.0, 0.0], [1.0, 0.0],
            [0.2, h*0.2], [0.8, h*0.2],
            [0.5, h*0.35],
            [0.1, h*0.1], [0.9, h*0.1],
            [0.35, h*0.45], [0.65, h*0.45],
            [0.5, h*0.75],
        ]),
        4: np.array([
            [0.5, 0.0],
            [0.0, 0.0], [1.0, 0.0],
            [0.2, h*0.2], [0.8, h*0.2],
            [0.5, h*0.4],
            [0.1, h*0.1], [0.9, h*0.1],
            [0.3, h*0.5], [0.7, h*0.5],
            [0.5, h*0.8],
        ]),
    }
    
    if pattern_id not in configs:
        pattern_id = 0
    config = configs[pattern_id].copy()
    
    if jitter > 0 and rng is not None:
        config = config + rng.normal(0, jitter, size=config.shape)
    
    config = _project_to_triangle(config.copy())
    return config


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    h = np.sqrt(3) / 2.0
    rng = np.random.RandomState(42)
    
    # Initialize with best pattern
    best_points = _make_config(0)
    best_area = _min_triangle_area(best_points)
    
    # Phase 1: Optimize each pattern with small perturbations
    for pat in range(5):
        for jitter in [0.0, 0.005, 0.01, 0.02]:
            if time.time() > deadline:
                break
            
            pts = _make_config(pat, rng=rng, jitter=jitter)
            flat_pts = pts.flatten()
            
            try:
                result = minimize(
                    _objective_flat,
                    flat_pts,
                    method='SLSQP',
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                opt_pts = result.x.reshape(N, 2)
                opt_pts = _project_to_triangle(opt_pts.copy())
                opt_area = _min_triangle_area(opt_pts)
                if opt_area > best_area:
                    best_area = opt_area
                    best_points = opt_pts.copy()
            except Exception:
                pass
    
    # Phase 2: Basin hopping
    n_hops = 60
    for hop in range(n_hops):
        if time.time() > deadline:
            break
        
        scale = 0.03 + 0.02 * (1.0 - hop / n_hops)
        pts = best_points + rng.normal(0, scale, size=(N, 2))
        pts = _project_to_triangle(pts.copy())
        flat_pts = pts.flatten()
        
        try:
            result = minimize(
                _objective_flat,
                flat_pts,
                method='SLSQP',
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            opt_pts = result.x.reshape(N, 2)
            opt_pts = _project_to_triangle(opt_pts.copy())
            opt_area = _min_triangle_area(opt_pts)
            if opt_area > best_area:
                best_area = opt_area
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Phase 3: Final polish
    if time.time() < deadline - 3:
        flat_pts = best_points.flatten()
        try:
            result = minimize(
                _objective_flat,
                flat_pts,
                method='SLSQP',
                options={'maxiter': 500, 'ftol': 1e-13}
            )
            opt_pts = result.x.reshape(N, 2)
            opt_pts = _project_to_triangle(opt_pts.copy())
            opt_area = _min_triangle_area(opt_pts)
            if opt_area > best_area:
                best_area = opt_area
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    return best_points
