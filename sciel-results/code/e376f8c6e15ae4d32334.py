"""Heilbronn triangles: maximize the smallest triangle area among 11 points."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

# Precompute all triplets
_TRIPLETS = list(combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)

DEADLINE = 145  # Safety margin for 150s limit


def _bary_to_cart(bc):
    """Convert barycentric (b, c) to cartesian. a = 1-b-c.
    Vertices: A=(0,0), B=(1,0), C=(0.5, S3H)
    cart = b*B + c*C = (b + 0.5*c, S3H*c)
    """
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = S3H * c
    return np.column_stack([x, y])


def _cart_to_bary(pts):
    """Convert cartesian to barycentric (b, c). a = 1-b-c."""
    x = pts[:, 0]
    y = pts[:, 1]
    c = y / S3H
    b = x - 0.5 * c
    return np.column_stack([b, c])


def _project_bary_to_valid(bc):
    """Project barycentric coordinates to valid triangle (a>=0, b>=0, c>=0, a+b+c=1)."""
    b = np.clip(bc[:, 0], 0.0, 1.0)
    c = np.clip(bc[:, 1], 0.0, 1.0)
    # Ensure a = 1 - b - c >= 0
    a = 1.0 - b - c
    mask = a < 0
    if np.any(mask):
        # Scale b and c to make a = 0
        scale = 1.0 / (b[mask] + c[mask])
        b[mask] *= scale
        c[mask] *= scale
    return np.column_stack([b, c])


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


def _all_triangle_areas(pts):
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return np.abs(cross) * 0.5


def _soft_min_area(pts, beta):
    areas = _all_triangle_areas(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_a - log_sum / beta


def _objective_bary(bc_flat, beta=50.0):
    """Objective in barycentric space: minimize -soft_min_area."""
    bc = bc_flat.reshape(-1, 2)
    bc = _project_bary_to_valid(bc)
    pts = _bary_to_cart(bc)
    return -_soft_min_area(pts, beta)


def _objective_bary_hard(bc_flat):
    """Objective in barycentric space: minimize -min_area (hard min)."""
    bc = bc_flat.reshape(-1, 2)
    bc = _project_bary_to_valid(bc)
    pts = _bary_to_cart(bc)
    return -_min_triangle_area(pts)


def _push_apart(pts, areas, active_indices, strength=0.01):
    """Push apart points that form active (small) triangles."""
    new_pts = pts.copy()
    for idx in active_indices:
        i, j, k = _TRIPLETS[idx]
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        
        # Compute vectors
        v1 = pj - pi
        v2 = pk - pi
        
        # Area of triangle
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        area = abs(cross) * 0.5
        
        if area < 1e-10:
            continue
            
        # Direction to increase area: push points apart along the altitude
        # For triangle (pi, pj, pk), increasing area means moving points away from each other
        
        # Simple heuristic: move each point away from the centroid
        centroid = (pi + pj + pk) / 3.0
        
        # Move pi away from centroid
        dir_i = pi - centroid
        norm_i = np.linalg.norm(dir_i)
        if norm_i > 1e-10:
            dir_i /= norm_i
            new_pts[i] += strength * dir_i
        
        # Move pj away from centroid
        dir_j = pj - centroid
        norm_j = np.linalg.norm(dir_j)
        if norm_j > 1e-10:
            dir_j /= norm_j
            new_pts[j] += strength * dir_j
        
        # Move pk away from centroid
        dir_k = pk - centroid
        norm_k = np.linalg.norm(dir_k)
        if norm_k > 1e-10:
            dir_k /= norm_k
            new_pts[k] += strength * dir_k
    
    return new_pts


def _project_to_triangle(pts):
    """Vectorized projection into equilateral triangle."""
    y = np.clip(pts[:, 1], 0.0, S3H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(pts[:, 0], x_min, x_max)
    return np.column_stack([x, y])


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin=5):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts.copy())
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # Symmetric initial configuration
    init = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.2, 0.1],
    ], dtype=float)
    _update_best(init)

    # Phase 1: Global Search using barycentric coordinates with multiple restarts
    if _elapsed_ok(60):
        # Use barycentric representation for natural constraint handling
        for restart in range(8):
            if not _elapsed_ok(10):
                break
            
            # Random initial barycentric coordinates
            if restart == 0:
                bc0 = _cart_to_bary(init)
            else:
                bc0 = rng.uniform(0, 1, size=(N, 2))
                bc0 = _project_bary_to_valid(bc0)
            
            # Optimize in barycentric space
            def bary_obj(bc_flat):
                return _objective_bary(bc_flat, beta=30.0)
            
            x0 = bc0.flatten()
            
            try:
                res = minimize(
                    bary_obj,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                
                bc_opt = _project_bary_to_valid(res.x.reshape(-1, 2))
                pts_opt = _bary_to_cart(bc_opt)
                _update_best(pts_opt)
                
                # Refine with hard min objective
                def bary_obj_hard(bc_flat):
                    return _objective_bary_hard(bc_flat)
                
                res2 = minimize(
                    bary_obj_hard,
                    res.x,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                
                bc_opt2 = _project_bary_to_valid(res2.x.reshape(-1, 2))
                pts_opt2 = _bary_to_cart(bc_opt2)
                _update_best(pts_opt2)
                
            except Exception:
                pass

    # Phase 2: Local Refinement using Active Set + Gradient-based push
    if best_pts is not None and _elapsed_ok(20):
        current_pts = best_pts.copy()
        
        for restart in range(5):
            if not _elapsed_ok(10):
                break
            
            if restart > 0:
                current_pts = _project_to_triangle(current_pts + rng.normal(0, 0.005, size=(N, 2)))
            
            # Iterative Active Set Optimization with push-apart
            for iter_count in range(15):
                if not _elapsed_ok(5):
                    break
                
                areas = _all_triangle_areas(current_pts)
                min_area = np.min(areas)
                
                # Identify active triplets
                area_range = np.max(areas) - np.min(areas)
                tol = 0.03 * area_range + 1e-6
                active_mask = areas < (min_area + tol)
                active_indices = np.where(active_mask)[0]
                
                if len(active_indices) == 0:
                    active_indices = np.array([np.argmin(areas)])
                
                # Apply push-apart
                pushed_pts = _push_apart(current_pts, areas, active_indices, strength=0.005)
                pushed_pts = _project_to_triangle(pushed_pts)
                
                # Optimize with SLSQP on the active set
                def active_obj(flat):
                    pts = flat.reshape(-1, 2)
                    a = _all_triangle_areas(pts)
                    return -np.min(a[active_indices])
                
                constraints = []
                for i in range(N):
                    constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
                    constraints.append({'type': 'ineq', 'fun': lambda x, i=i: S3H - x[2*i+1]})
                    constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*i+1]/S3})
                    constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i+1]/S3 - x[2*i]})

                x0 = pushed_pts.flatten()
                
                try:
                    res = minimize(
                        active_obj, 
                        x0, 
                        method='SLSQP', 
                        bounds=[(0, 1), (0, S3H)] * N,
                        constraints=constraints,
                        options={'maxiter': 100, 'ftol': 1e-12}
                    )
                    if res.success:
                        new_pts = _project_to_triangle(res.x.reshape(-1, 2))
                        new_area = _min_triangle_area(new_pts)
                        if new_area > min_area - 1e-9:
                            current_pts = new_pts
                            _update_best(new_pts)
                        else:
                            break
                    else:
                        break
                except Exception:
                    break

    # Phase 3: Final High-Precision Polish in barycentric space
    if best_pts is not None and _elapsed_ok(5):
        bc0 = _cart_to_bary(best_pts)
        
        def bary_obj_final(bc_flat):
            return _objective_bary_hard(bc_flat)
        
        x0 = bc0.flatten()
        try:
            res = minimize(
                bary_obj_final, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            bc_opt = _project_bary_to_valid(res.x.reshape(-1, 2))
            pts_opt = _bary_to_cart(bc_opt)
            _update_best(pts_opt)
        except Exception:
            pass

    if best_pts is None:
        pts = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, S3H],
            [0.5, 0.0],
            [0.75, S3H / 2],
            [0.25, S3H / 2],
            [1/3, S3H/3],
            [0.333, 0.0],
            [0.667, 0.0],
            [0.5, S3H/3],
            [0.2, 0.1],
        ], dtype=float)
        return _project_to_triangle(pts)

    return _project_to_triangle(best_pts)
