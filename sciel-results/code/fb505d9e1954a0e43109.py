"""
Heilbronn triangles: maximize the smallest triangle area among 11 points.
Strategy: SLSQP with explicit 165 triangle-area constraints + Nelder-Mead polish + multi-start.
"""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
M = len(_TRIPLETS)  # 165
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

DEADLINE = 148  # Safety margin from 150s limit


def _bary_to_cart(bary):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    bary: shape (N, 2) where u, v >= 0 and u+v <= 1.
    P = u*V0 + v*V1 + (1-u-v)*V2
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = u * V0[0] + v * V1[0] + w * V2[0]
    y = u * V0[1] + v * V1[1] + w * V2[1]
    return np.column_stack([x, y])


def _cart_to_bary(pts):
    """
    Convert Cartesian to barycentric (u, v).
    """
    u = (S3H - pts[:, 1]) / S3H
    v = 1.0 - 2.0 * pts[:, 0]
    return np.column_stack([u, v])


def _min_triangle_area_cart(pts):
    """Compute the minimum area of any triangle formed by three points (vectorized)."""
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


def _all_triangle_areas_cart(pts):
    """Compute all triangle areas (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return np.abs(cross) * 0.5


def _project_bary_to_valid(bary):
    """Ensure barycentric coordinates are within the triangle."""
    u = np.clip(bary[:, 0], 0, 1)
    v = np.clip(bary[:, 1], 0, 1)
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        scale = 1.0 / sum_uv[mask]
        u[mask] *= scale
        v[mask] *= scale
    return np.column_stack([u, v])


def _slsqp_objective(x, t_min):
    """
    Objective for SLSQP: minimize sum of squared violations of area >= t_min.
    x: shape (2*N,) barycentric coordinates flattened.
    """
    b = x.reshape(N, 2)
    b = _project_bary_to_valid(b)
    cart = _bary_to_cart(b)
    areas = _all_triangle_areas_cart(cart)
    violations = np.maximum(0, t_min - areas)
    return np.sum(violations ** 2)


def _slsqp_constraints(x, t_min):
    """
    Constraints for SLSQP: each triangle area >= t_min.
    Returns list of constraint dicts.
    """
    b = x.reshape(N, 2)
    b = _project_bary_to_valid(b)
    cart = _bary_to_cart(b)
    areas = _all_triangle_areas_cart(cart)
    # constraints: areas[i] - t_min >= 0
    return list(areas - t_min)


def _cartesian_objective(x):
    """
    Objective for Nelder-Mead in Cartesian: negative min triangle area.
    x: shape (2*N,) Cartesian coordinates flattened.
    """
    pts = x.reshape(N, 2)
    # Project to triangle if needed
    # Check if points are inside triangle
    # For simplicity, just compute; NM will stay near valid region
    return -_min_triangle_area_cart(pts)


def _bary_objective(x):
    """
    Objective for Nelder-Mead in barycentric: negative min triangle area.
    x: shape (2*N,) barycentric coordinates flattened.
    """
    b = x.reshape(N, 2)
    b = _project_bary_to_valid(b)
    cart = _bary_to_cart(b)
    return -_min_triangle_area_cart(cart)


def _make_symmetric_init():
    """Create a symmetric initial configuration."""
    # Barycentric (u, v) where P = u*V0 + v*V1 + (1-u-v)*V2
    # V0=(1,0,0) -> (1,0), V1=(0,1,0) -> (0,1), V2=(0,0,1) -> (0,0)
    # Symmetric placement: corners, edge midpoints, center, and 4 more
    bary = np.array([
        [1.0, 0.0],   # V0
        [0.0, 1.0],   # V1
        [0.0, 0.0],   # V2
        [0.5, 0.5],   # Mid V0-V1
        [0.0, 0.5],   # Mid V1-V2
        [0.5, 0.0],   # Mid V2-V0
        [1/3, 1/3],   # Center
        [0.2, 0.3],   # Extra
        [0.3, 0.2],   # Extra
        [0.4, 0.1],   # Extra
        [0.1, 0.4],   # Extra
    ])
    return bary


def _random_init(rng):
    """Generate random initial configuration."""
    u = rng.random(N)
    v = rng.random(N)
    mask = u + v > 1
    if np.any(mask):
        scale = 1.0 / (u[mask] + v[mask])
        u[mask] *= scale
        v[mask] *= scale
    return np.column_stack([u, v])


def _slsqp_optimize(bary_init, t_min, maxiter=500):
    """
    Run SLSQP to find configuration where all triangles have area >= t_min.
    Returns (success, bary_result).
    """
    x0 = bary_init.flatten()
    
    def obj(x):
        return _slsqp_objective(x, t_min)
    
    # Use inequality constraints: area_i - t_min >= 0
    # We'll use the constraint function form
    cons = [{'type': 'ineq', 'fun': lambda x, i=i: _slsqp_constraints(x, t_min)[i]} for i in range(M)]
    
    # Bounds: barycentric coords
    bounds = [(0, 1)] * (2 * N)
    
    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': 1e-10})
        bary_res = _project_bary_to_valid(res.x.reshape(N, 2))
        return res.success, bary_res
    except Exception:
        return False, bary_init


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_bary = None
    best_area = 0.0
    best_cart = None

    def _elapsed_ok(margin=5):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(bary):
        nonlocal best_bary, best_area, best_cart
        bary = _project_bary_to_valid(bary)
        cart = _bary_to_cart(bary)
        area = _min_triangle_area_cart(cart)
        if area > best_area:
            best_area = area
            best_bary = bary.copy()
            best_cart = cart.copy()

    # Initial configurations
    init_list = []
    
    # Symmetric start
    init_list.append(_make_symmetric_init())
    
    # Random starts
    for _ in range(30):
        init_list.append(_random_init(rng))
    
    # Multi-start SLSQP with increasing t_min
    # Strategy: start with a low t_min, find feasible solution, then increase
    
    # First, find a good starting t_min from random configs
    for bary0 in init_list[:10]:
        if not _elapsed_ok(30):
            break
        cart = _bary_to_cart(bary0)
        area = _min_triangle_area_cart(cart)
        if area > best_area:
            best_area = area
            best_bary = bary0.copy()
            best_cart = cart.copy()
    
    # Now iteratively increase t_min using SLSQP
    if best_bary is not None:
        current_t = best_area * 0.95  # Start slightly below current best
        max_t_attempts = 20
        
        for attempt in range(max_t_attempts):
            if not _elapsed_ok(20):
                break
            
            target_t = current_t * 1.05  # Try to increase by 5%
            
            # Try SLSQP from current best
            success, bary_res = _slsqp_optimize(best_bary, target_t, maxiter=300)
            
            if success:
                cart_res = _bary_to_cart(bary_res)
                actual_min = _min_triangle_area_cart(cart_res)
                if actual_min >= target_t * 0.99:
                    # Success! Update best and increase target
                    _update_best(bary_res)
                    current_t = actual_min
                else:
                    # Partial success, use actual min
                    _update_best(bary_res)
                    current_t = actual_min
            else:
                # Try from a few random restarts
                found = False
                for _ in range(3):
                    if not _elapsed_ok(15):
                        break
                    bary_rand = _random_init(rng)
                    success2, bary_res2 = _slsqp_optimize(bary_rand, target_t, maxiter=200)
                    if success2:
                        cart_res2 = _bary_to_cart(bary_res2)
                        actual_min2 = _min_triangle_area_cart(cart_res2)
                        if actual_min2 >= target_t * 0.99:
                            _update_best(bary_res2)
                            current_t = actual_min2
                            found = True
                            break
                        else:
                            _update_best(bary_res2)
                            current_t = max(current_t, actual_min2)
                
                if not found:
                    # Can't increase, try Nelder-Mead polish on best
                    break
        
        # Nelder-Mead polish on the best found
        if best_bary is not None and _elapsed_ok(10):
            x0 = best_bary.flatten()
            try:
                res_nm = minimize(_bary_objective, x0, method='Nelder-Mead',
                                  options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
                bary_nm = _project_bary_to_valid(res_nm.x.reshape(N, 2))
                _update_best(bary_nm)
            except Exception:
                pass
        
        # Additional SLSQP attempts from different starting points with high t_min
        if _elapsed_ok(20):
            high_t = best_area * 1.02
            for _ in range(5):
                if not _elapsed_ok(15):
                    break
                bary_start = _random_init(rng)
                # Mix with current best
                bary_start = 0.5 * bary_start + 0.5 * best_bary
                bary_start = _project_bary_to_valid(bary_start)
                
                success, bary_res = _slsqp_optimize(bary_start, high_t, maxiter=400)
                if success:
                    _update_best(bary_res)
                    # If we improved, try even higher
                    if best_area > high_t:
                        high_t = best_area * 1.02
        
        # Final Nelder-Mead polish
        if best_bary is not None and _elapsed_ok(8):
            x0 = best_bary.flatten()
            try:
                res_final = minimize(_bary_objective, x0, method='Nelder-Mead',
                                     options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13})
                bary_final = _project_bary_to_valid(res_final.x.reshape(N, 2))
                _update_best(bary_final)
            except Exception:
                pass
        
        # One more SLSQP round from the polished best
        if best_bary is not None and _elapsed_ok(10):
            t_target = best_area * 1.01
            success, bary_res = _slsqp_optimize(best_bary, t_target, maxiter=300)
            if success:
                _update_best(bary_res)
            
            # Polish again
            if _elapsed_ok(5):
                x0 = best_bary.flatten()
                try:
                    res_p = minimize(_bary_objective, x0, method='Nelder-Mead',
                                     options={'maxiter': 1000, 'xatol': 1e-11, 'fatol': 1e-13})
                    bary_p = _project_bary_to_valid(res_p.x.reshape(N, 2))
                    _update_best(bary_p)
                except Exception:
                    pass

    if best_cart is None:
        # Fallback
        bary = _make_symmetric_init()
        best_cart = _bary_to_cart(bary)

    return best_cart
