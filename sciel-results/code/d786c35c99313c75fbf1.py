"""
Heilbronn triangles: maximize the smallest triangle area among 11 points.
Strategy: Largest Empty Triangle heuristic + Simulated Annealing + Nelder-Mead Polish.
"""
import numpy as np
import time
from scipy.optimize import minimize
import itertools
import math

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIPLETS = len(_TRIPLETS)

DEADLINE = 145  # Safety margin from 150s limit

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
    areas = np.abs(cross) * 0.5
    return areas

def _min_triangle_area_cart(pts):
    """Compute the minimum area of any triangle formed by three points."""
    return np.min(_all_triangle_areas_cart(pts))

def _soft_min_area_cart(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    areas = _all_triangle_areas_cart(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min

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

def _largest_empty_triangle_move(pts, bary):
    """
    Find the minimum-area triangle, compute centroid of convex hull of other points,
    and replace the point in the min triangle that is closest to that centroid.
    Returns new bary coordinates.
    """
    areas = _all_triangle_areas_cart(pts)
    min_idx = np.argmin(areas)
    min_tri = _TRIPLETS[min_idx]
    
    # Points not in the minimum triangle
    other_indices = [i for i in range(N) if i not in min_tri]
    other_pts = pts[other_indices]
    
    # Compute centroid of other points (approximation of convex hull centroid)
    centroid = np.mean(other_pts, axis=0)
    
    # Project centroid to barycentric and ensure valid
    centroid_bary = _cart_to_bary(centroid.reshape(1, 2))[0]
    centroid_bary = np.clip(centroid_bary, 0, 1)
    s = centroid_bary[0] + centroid_bary[1]
    if s > 1:
        centroid_bary /= s
    
    # Find which point in the min triangle to replace: the one closest to centroid
    # in Cartesian space, but we want to replace the point that, when moved,
    # best increases the minimum area. Heuristic: replace the point closest to centroid.
    distances = [np.linalg.norm(pts[i] - centroid) for i in min_tri]
    replace_idx = min_tri[np.argmin(distances)]
    
    # New bary
    new_bary = bary.copy()
    new_bary[replace_idx] = centroid_bary
    return new_bary

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_bary = None
    best_area = 0.0
    best_cart = None

    def _elapsed_ok(margin):
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
    init_bary_list = []
    
    # Base symmetric configuration
    base_sym = np.array([
        [1.0, 0.0],   # V0
        [0.0, 1.0],   # V1
        [0.0, 0.0],   # V2
        [0.5, 0.5],   # Mid V0-V1
        [0.0, 0.5],   # Mid V1-V2
        [0.5, 0.0],   # Mid V2-V0
        [1/3, 1/3],   # Center
        [0.2, 0.2],
        [0.3, 0.3],
        [0.4, 0.2],
        [0.2, 0.4],
    ])
    init_bary_list.append(base_sym)

    # Generate random initial configurations
    for _ in range(30):
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        if np.any(mask):
            scale = 1.0 / (u[mask] + v[mask])
            u[mask] *= scale
            v[mask] *= scale
        init_bary_list.append(np.column_stack([u, v]))

    # Add some "near-boundary" configurations
    for _ in range(10):
        u = rng.random(N)
        v = rng.random(N)
        for i in range(N):
            if rng.random() < 0.3:
                edge = rng.integers(0, 3)
                if edge == 0:
                    u[i] = 0
                elif edge == 1:
                    v[i] = 0
                else:
                    s = u[i] + v[i]
                    if s > 0:
                        u[i] /= s
                        v[i] /= s
                    else:
                        u[i] = 0.5
                        v[i] = 0.5
        init_bary_list.append(np.column_stack([u, v]))

    # Optimization Loop
    for bary0 in init_bary_list:
        if not _elapsed_ok(20):
            break
        
        current_bary = bary0.copy()
        current_bary = _project_bary_to_valid(current_bary)
        _update_best(current_bary)
        
        # Phase 1: Largest Empty Triangle heuristic iterations
        if _elapsed_ok(15):
            for _ in range(50):
                if not _elapsed_ok(5):
                    break
                cart = _bary_to_cart(current_bary)
                new_bary = _largest_empty_triangle_move(cart, current_bary)
                new_bary = _project_bary_to_valid(new_bary)
                _update_best(new_bary)
                current_bary = new_bary

        # Phase 2: Simulated Annealing
        if _elapsed_ok(20):
            current_bary = best_bary.copy() if best_bary is not None else bary0.copy()
            current_bary = _project_bary_to_valid(current_bary)
            
            T_start = 0.05
            T_end = 0.001
            n_sa_iter = 2000
            
            for it in range(n_sa_iter):
                if not _elapsed_ok(5):
                    break
                # Temperature schedule
                T = T_start * (T_end / T_start) ** (it / n_sa_iter)
                
                # Generate perturbation
                perturbation_scale = 0.05 * (T / T_start) ** 0.5
                delta = rng.normal(0, perturbation_scale, size=(N, 2))
                new_bary = current_bary + delta
                new_bary = _project_bary_to_valid(new_bary)
                
                # Evaluate
                new_cart = _bary_to_cart(new_bary)
                new_area = _min_triangle_area_cart(new_cart)
                cur_cart = _bary_to_cart(current_bary)
                cur_area = _min_triangle_area_cart(cur_cart)
                
                delta_f = new_area - cur_area
                
                if delta_f > 0 or rng.random() < np.exp(delta_f / T):
                    current_bary = new_bary
                    _update_best(current_bary)

        # Phase 3: Nelder-Mead polish on hard minimum
        if _elapsed_ok(10):
            nm_start = best_bary.copy() if best_bary is not None else bary0.copy()
            
            def nm_objective(x):
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                return -_min_triangle_area_cart(cart)
            
            try:
                res_nm = minimize(nm_objective, nm_start.flatten(), method='Nelder-Mead',
                                  options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11,
                                           'adaptive': True})
                current_bary = res_nm.x.reshape(N, 2)
                current_bary = _project_bary_to_valid(current_bary)
                _update_best(current_bary)
            except Exception:
                pass

        # Phase 4: More L-BFGS-B with soft min for smooth optimization
        if _elapsed_ok(10):
            for beta in [100.0, 500.0, 2000.0]:
                if not _elapsed_ok(5):
                    break
                
                def objective(x):
                    b = x.reshape(N, 2)
                    b = _project_bary_to_valid(b)
                    cart = _bary_to_cart(b)
                    return -_soft_min_area_cart(cart, beta)
                
                x0 = (best_bary if best_bary is not None else bary0).flatten()
                
                try:
                    res = minimize(objective, x0, method='L-BFGS-B',
                                   options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12})
                    current_bary = res.x.reshape(N, 2)
                    current_bary = _project_bary_to_valid(current_bary)
                    _update_best(current_bary)
                except Exception:
                    pass

    # Final polish on the absolute best found
    if best_bary is not None and _elapsed_ok(8):
        def final_nm_obj(x):
            b = x.reshape(N, 2)
            b = _project_bary_to_valid(b)
            cart = _bary_to_cart(b)
            return -_min_triangle_area_cart(cart)
        
        try:
            res_final = minimize(final_nm_obj, best_bary.flatten(), method='Nelder-Mead',
                                 options={'maxiter': 1000, 'xatol': 1e-11, 'fatol': 1e-13,
                                          'adaptive': True})
            final_bary = res_final.x.reshape(N, 2)
            final_bary = _project_bary_to_valid(final_bary)
            _update_best(final_bary)
        except Exception:
            pass
        
        # Second polish with even tighter tolerance if time permits
        if _elapsed_ok(5):
            try:
                res_final2 = minimize(final_nm_obj, best_bary.flatten(), method='Nelder-Mead',
                                      options={'maxiter': 500, 'xatol': 1e-13, 'fatol': 1e-15,
                                               'adaptive': True})
                final_bary2 = res_final2.x.reshape(N, 2)
                final_bary2 = _project_bary_to_valid(final_bary2)
                _update_best(final_bary2)
            except Exception:
                pass

    if best_cart is None:
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        bary = np.column_stack([u, v])
        best_cart = _bary_to_cart(bary)
        
    return best_cart
