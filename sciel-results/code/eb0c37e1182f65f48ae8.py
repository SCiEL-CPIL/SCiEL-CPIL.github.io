"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle.

Combines barycentric-coordinate trust-region optimization (structural idea) with
multi-start refinement machinery from the inspirations.
"""
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2.0

# Precompute all C(11,3)=165 triangle index triples
_combos = list(combinations(range(N), 3))
_ci = np.array([c[0] for c in _combos])
_cj = np.array([c[1] for c in _combos])
_ck = np.array([c[2] for c in _combos])


def _bary_to_cart(bary):
    """Convert barycentric coordinates (n, 3) to Cartesian (n, 2).
    Triangle vertices: A=(0,0), B=(1,0), C=(0.5, H)"""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = H * bary[:, 2]
    return np.column_stack([x, y])


def _cart_to_bary(pts):
    """Convert Cartesian (n, 2) to barycentric (n, 3) coordinates."""
    x = pts[:, 0]
    y = pts[:, 1]
    lam3 = y / H
    lam2 = x - 0.5 * lam3
    lam1 = 1.0 - lam2 - lam3
    return np.column_stack([lam1, lam2, lam3])


def _min_triangle_area(points):
    """Compute minimum area of all C(11,3) triangles using vectorized ops."""
    pi = points[_ci]
    pj = points[_cj]
    pk = points[_ck]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return float(np.min(areas))


def _project_to_triangle(pts):
    """Project points to stay inside the equilateral triangle (vectorized)."""
    pts = pts.copy()
    x = pts[:, 0]
    y = pts[:, 1]
    x = np.clip(x, 0.0, 1.0)
    y = np.maximum(y, 0.0)
    upper = np.minimum(H * x, H * (1.0 - x))
    y = np.minimum(y, upper)
    pts[:, 0] = x
    pts[:, 1] = y
    return pts


def _neg_min_area_cart(flat_pts):
    """Objective in Cartesian: negative of minimum triangle area."""
    pts = flat_pts.reshape(N, 2)
    pts = _project_to_triangle(pts)
    return -_min_triangle_area(pts)


def _neg_min_area_bary(flat_bary):
    """Objective in barycentric coords: negative min area.
    Barycentric coords naturally satisfy sum=1, non-negativity via bounds."""
    bary = flat_bary.reshape(N, 3)
    # Normalize to ensure sum=1 (handles small numerical drift)
    row_sums = bary.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-15)
    bary = bary / row_sums
    # Clamp to [0,1]
    bary = np.clip(bary, 0.0, 1.0)
    # Re-normalize after clamping
    row_sums = bary.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-15)
    bary = bary / row_sums
    
    pts = _bary_to_cart(bary)
    return -_min_triangle_area(pts)


def _initial_config_symmetric():
    """Symmetric arrangement based on literature for 11 points in equilateral triangle."""
    base_config = np.array([
        [0.5, 0.0],        # bottom center
        [0.0, 0.0],        # bottom left
        [1.0, 0.0],        # bottom right
        [0.25, H * 0.25],  # left mid
        [0.75, H * 0.25],  # right mid
        [0.5, H * 0.5],    # center
        [0.125, H * 0.125],# lower left
        [0.875, H * 0.125],# lower right
        [0.375, H * 0.375],# mid left
        [0.625, H * 0.375],# mid right
        [0.5, H * 0.875],  # top
    ])
    return _project_to_triangle(base_config)


def _initial_config_grid():
    """Structured grid arrangement within the triangle."""
    pts = []
    # Layer 0 (bottom): 3 points along the base
    for i in range(3):
        pts.append([0.15 + 0.35 * i, 0.05])
    # Layer 1: 4 points
    for i in range(4):
        pts.append([0.1 + 0.25 * i, 0.22])
    # Layer 2: 3 points
    for i in range(3):
        pts.append([0.2 + 0.3 * i, 0.42])
    # Layer 3 (top): 1 point
    pts.append([0.5, 0.55])
    pts = np.array(pts)
    return _project_to_triangle(pts)


def _local_optimize_bary_trf(pts, max_time=40.0):
    """Run local optimization using Trust Region Reflective on barycentric coordinates.
    
    This is the structural change: by working in barycentric coordinates with
    bounds [0,1] on each coordinate (with implicit sum-to-1 normalization),
    the TRF algorithm naturally handles the convex constraint that points
    must lie inside the triangle, without needing explicit projection at
    every evaluation step.
    """
    best_pts = pts.copy()
    best_score = _min_triangle_area(pts)
    t0 = time.time()
    
    # Convert to barycentric
    bary = _cart_to_bary(pts)
    # Clamp to valid range
    bary = np.clip(bary, 0.0, 1.0)
    row_sums = bary.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-15)
    bary = bary / row_sums
    
    flat_bary = bary.flatten()
    
    # Bounds: each barycentric coordinate in [0, 1]
    bounds = [(0.0, 1.0)] * (3 * N)
    
    try:
        result = minimize(
            _neg_min_area_bary,
            flat_bary,
            method='trust-constr',
            bounds=bounds,
            options={'maxiter': 300, 'gtol': 1e-8, 'xtol': 1e-10}
        )
        
        if time.time() - t0 < max_time:
            opt_bary = result.x.reshape(N, 3)
            # Normalize
            opt_bary = np.clip(opt_bary, 0.0, 1.0)
            row_sums = opt_bary.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-15)
            opt_bary = opt_bary / row_sums
            
            opt_pts = _bary_to_cart(opt_bary)
            opt_pts = _project_to_triangle(opt_pts)
            opt_score = _min_triangle_area(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
    except Exception:
        pass
    
    return best_pts, best_score


def _local_optimize_slsqp(pts, max_time=40.0):
    """Run local optimization using SLSQP in Cartesian coordinates."""
    best_pts = pts.copy()
    best_score = _min_triangle_area(pts)
    t0 = time.time()
    
    flat_pts = pts.flatten()
    
    try:
        result = minimize(
            _neg_min_area_cart, flat_pts, method='SLSQP',
            options={'maxiter': 300, 'ftol': 1e-12}
        )
        
        if time.time() - t0 < max_time:
            opt_pts = result.x.reshape(N, 2)
            opt_pts = _project_to_triangle(opt_pts)
            opt_score = _min_triangle_area(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
    except Exception:
        pass
    
    return best_pts, best_score


def _local_optimize_nm(pts, max_time=40.0):
    """Run local optimization using Nelder-Mead starting from pts."""
    best_pts = pts.copy()
    best_score = _min_triangle_area(pts)
    t0 = time.time()
    
    flat_pts = pts.flatten()
    
    try:
        result = minimize(
            _neg_min_area_cart, flat_pts, method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        
        if time.time() - t0 < max_time:
            opt_pts = result.x.reshape(N, 2)
            opt_pts = _project_to_triangle(opt_pts)
            opt_score = _min_triangle_area(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
    except Exception:
        pass
    
    return best_pts, best_score


def _max_min_distance_init(rng, n_iter=300):
    """Initialize points by maximizing minimum pairwise distance (greedy + random restarts)."""
    best_pts = None
    best_min_dist = 0.0
    
    for _ in range(n_iter):
        pts = [rng.uniform(0, 1, 2)]
        pts = [_project_to_triangle(np.array(pts)).flatten()]
        
        for i in range(1, N):
            best_candidate = None
            best_dist = 0.0
            
            for _ in range(200):
                candidate = rng.uniform(0, 1, 2)
                candidate = _project_to_triangle(np.array([candidate])).flatten()
                
                dists = np.linalg.norm(np.array(pts) - candidate, axis=1)
                min_dist = np.min(dists)
                
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_candidate = candidate
            
            pts.append(best_candidate)
        
        pts = np.array(pts)
        
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_pts = pts.copy()
    
    return best_pts


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Multiple diverse initializations
    init_configs = [
        _initial_config_symmetric(),
        _initial_config_grid(),
        _max_min_distance_init(rng, n_iter=200),
    ]
    
    best_points = init_configs[0].copy()
    best_score = _min_triangle_area(best_points)
    
    for cfg in init_configs[1:]:
        s = _min_triangle_area(cfg)
        if s > best_score:
            best_score = s
            best_points = cfg.copy()
    
    # Phase 2: Trust-constrained barycentric optimization from each init
    for cfg in init_configs:
        if time.time() > deadline - 30:
            break
        opt_pts, opt_score = _local_optimize_bary_trf(cfg, max_time=25.0)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts.copy()
    
    # Phase 3: SLSQP polish from best
    if time.time() < deadline - 20:
        opt_pts, opt_score = _local_optimize_slsqp(best_points, max_time=20.0)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts.copy()
    
    # Phase 4: Random restarts with barycentric TRF
    n_restarts = 5
    for i in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        # Perturbation in Cartesian
        perturbed = best_points + rng.normal(0, 0.04, size=(N, 2))
        perturbed = _project_to_triangle(perturbed)
        
        # Optimize in barycentric
        opt_pts, opt_score = _local_optimize_bary_trf(perturbed, max_time=15.0)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts.copy()
    
    # Phase 5: Final SLSQP polish
    if time.time() < deadline - 5:
        opt_pts, opt_score = _local_optimize_slsqp(best_points, max_time=10.0)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts.copy()
    
    # Ensure final result is valid
    best_points = _project_to_triangle(best_points)
    
    return best_points
