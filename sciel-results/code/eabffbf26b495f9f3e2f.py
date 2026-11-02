"""Heilbronn 11 triangles: Multi-start Nelder-Mead optimization."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

DEADLINE = 145  # Safety margin from 150s limit

# Precompute all triplets for N=11 points
# There are C(11, 3) = 165 triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

# Barycentric coordinates: vertices of the triangle
# V0 = (0, 0), V1 = (1, 0), V2 = (0.5, sqrt(3)/2)
_V0 = np.array([0.0, 0.0])
_V1 = np.array([1.0, 0.0])
_V2 = np.array([0.5, S3H])


def _bary_to_cart(bary):
    """Convert barycentric coordinates to Cartesian coordinates.
    bary has shape (n, 3) with columns (lambda0, lambda1, lambda2) where each row sums to 1.
    """
    # bary @ [V0, V1, V2]^T
    return bary[:, 0:1] * _V0 + bary[:, 1:2] * _V1 + bary[:, 2:3] * _V2


def _cart_to_bary(pts):
    """Convert Cartesian coordinates to barycentric coordinates.
    pts has shape (n, 2).
    Returns barycentric coordinates with shape (n, 3) where each row sums to 1.
    """
    # Solve: p = l0*V0 + l1*V1 + l2*V2, l0+l1+l2=1
    # V0=(0,0), V1=(1,0), V2=(0.5, S3H)
    # x = l1*1 + l2*0.5
    # y = l2*S3H
    # So l2 = y/S3H, l1 = x - 0.5*l2, l0 = 1 - l1 - l2
    l2 = pts[:, 1] / S3H
    l1 = pts[:, 0] - 0.5 * l2
    l0 = 1.0 - l1 - l2
    bary = np.column_stack([l0, l1, l2])
    # Clip to ensure validity (handle floating point errors)
    bary = np.clip(bary, 0, 1)
    # Normalize to sum to 1
    bary /= bary.sum(axis=1, keepdims=True)
    return bary


def _project_to_triangle(pts):
    """
    Project points into the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    """
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _min_triangle_area(pts):
    """
    Compute the minimum area of any triangle formed by three points.
    """
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


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts)
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    def _generate_initial_configs():
        """Generate diverse starting points in Cartesian coordinates."""
        configs = []
        
        # 1. Symmetric base: Corners, midpoints, centroid, etc.
        base_sym_cart = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, S3H],
            [0.5, 0.0],
            [0.75, S3H / 2],
            [0.25, S3H / 2],
            [1/3, S3H/3],
            [1/6, S3H/6],
            [5/6, S3H/6],
            [1/3, 2*S3H/3],
            [2/3, 2*S3H/3],
        ])
        configs.append(base_sym_cart)

        # 2. Random valid Cartesian configurations
        for _ in range(29): # Total 30 configs
            # Generate random points in the triangle
            # Method: generate u, v in [0,1], if u+v>1 reflect
            pts = np.zeros((N, 2))
            for i in range(N):
                u = rng.random()
                v = rng.random()
                if u + v > 1:
                    u = 1 - u
                    v = 1 - v
                x = u + 0.5 * v
                y = S3H * v
                pts[i] = [x, y]
            configs.append(pts)
            
        # 3. Perturbed symmetric base
        for _ in range(5):
            perturbed = base_sym_cart + rng.normal(0, 0.05, size=(N, 2))
            perturbed = _project_to_triangle(perturbed)
            configs.append(perturbed)

        return configs

    def _nelder_mead_optimize(pts0, maxiter=500):
        """Run Nelder-Mead on the hard objective."""
        def nm_obj(x):
            p = x.reshape(-1, 2)
            p = _project_to_triangle(p)
            return -_min_triangle_area(p)
        
        try:
            res = minimize(
                nm_obj,
                pts0.flatten(),
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            p = res.x.reshape(-1, 2)
            p = _project_to_triangle(p)
            return p
        except Exception:
            return pts0

    # Phase 1: Multi-start Nelder-Mead
    configs = _generate_initial_configs()
    
    # Shuffle configs to ensure we don't waste time on bad ones first? 
    # No, the symmetric base is good, keep it first.
    
    for i, config in enumerate(configs):
        if not _elapsed_ok(10):
            break
            
        # Run Nelder-Mead
        # Reduce maxiter if we are running out of time
        time_left = DEADLINE - (time.time() - start_time)
        # Heuristic: ~10-20ms per NM iteration? 22 dims.
        # Let's cap maxiter based on time left.
        # If time_left < 5s, break.
        
        maxiter = 1000 if time_left > 30 else 500
        p_opt = _nelder_mead_optimize(config, maxiter=maxiter)
        _update_best(p_opt)
        
        # Check time
        if not _elapsed_ok(5):
            break

    # Phase 2: Final fine polish on the absolute best
    if best_pts is not None and _elapsed_ok(10):
        def nm_obj_final(x):
            p = x.reshape(-1, 2)
            p = _project_to_triangle(p)
            return -_min_triangle_area(p)
        
        try:
            res = minimize(
                nm_obj_final,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            p = res.x.reshape(-1, 2)
            p = _project_to_triangle(p)
            _update_best(p)
        except Exception:
            pass

    if best_pts is None:
        # Absolute fallback: random points
        pts = rng.random((N, 2))
        for i in range(N):
            u, v = pts[i]
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = u + 0.5 * v
            y = S3H * v
            pts[i] = [x, y]
        best_pts = _project_to_triangle(pts)

    return best_pts
