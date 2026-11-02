"""Heilbronn triangles: maximize the smallest triangle area among 11 points."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
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

def _barycentric_to_cartesian(bc):
    """Convert barycentric (b, c) to cartesian (x, y). a = 1-b-c."""
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = S3H * c
    return np.column_stack([x, y])

def _cartesian_to_barycentric(pts):
    """Convert cartesian (x, y) to barycentric (b, c)."""
    c = pts[:, 1] / S3H
    b = pts[:, 0] - 0.5 * c
    return np.column_stack([b, c])

def _project_bary(bc):
    """Project barycentric coords to valid simplex: b>=0, c>=0, b+c<=1."""
    b = bc[:, 0].copy()
    c = bc[:, 1].copy()
    # Clip to non-negative
    b = np.clip(b, 0, None)
    c = np.clip(c, 0, None)
    # Clip to b+c <= 1
    s = b + c
    mask = s > 1
    if np.any(mask):
        b[mask] /= s[mask]
        c[mask] /= s[mask]
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

def _push_apart(bc, step=0.01, n_iter=50):
    """Directly push apart points that form small triangles.
    
    For each of the smallest triangles, compute the direction that increases
    the area and move the points in that direction.
    """
    bc = _project_bary(bc.copy())
    pts = _barycentric_to_cartesian(bc)
    areas = _all_triangle_areas(pts)
    min_area = np.min(areas)
    
    for _ in range(n_iter):
        areas = _all_triangle_areas(pts)
        min_area = np.min(areas)
        
        # Get the k smallest triangles
        k = min(10, _N_TRIP)
        small_idx = np.argsort(areas)[:k]
        
        improved = False
        for idx in small_idx:
            i, j, k_idx = _TRIPLETS[idx]
            pi = pts[i]
            pj = pts[j]
            pk = pts[k_idx]
            
            # Area = 0.5 * |cross(pj-pi, pk-pi)|
            v1 = pj - pi
            v2 = pk - pi
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            area = 0.5 * abs(cross)
            
            if area < min_area * 1.05:
                # Direction to increase area: move points apart
                # Gradient of area w.r.t. each point
                # d(area)/d(pi) = -0.5 * perp(v1-v2) ... let's use a simpler approach
                # Move i away from centroid of j,k
                # Move j away from centroid of i,k
                # Move k away from centroid of i,j
                
                centroid_jk = (pj + pk) / 2
                centroid_ik = (pi + pk) / 2
                centroid_ij = (pi + pj) / 2
                
                dir_i = pi - centroid_jk
                dir_j = pj - centroid_ik
                dir_k = pk - centroid_ij
                
                norm_i = np.linalg.norm(dir_i)
                norm_j = np.linalg.norm(dir_j)
                norm_k = np.linalg.norm(dir_k)
                
                if norm_i > 1e-10:
                    dir_i /= norm_i
                if norm_j > 1e-10:
                    dir_j /= norm_j
                if norm_k > 1e-10:
                    dir_k /= norm_k
                
                new_pts = pts.copy()
                new_pts[i] = pi + step * dir_i
                new_pts[j] = pj + step * dir_j
                new_pts[k_idx] = pk + step * dir_k
                
                # Project to triangle
                new_bc = _project_bary(_cartesian_to_barycentric(new_pts))
                new_pts = _barycentric_to_cartesian(new_bc)
                new_min = _min_triangle_area(new_pts)
                
                if new_min > min_area - 1e-12:
                    pts = new_pts
                    bc = new_bc
                    min_area = new_min
                    improved = True
        
        if not improved:
            break
    
    return bc, pts

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin=5):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # Symmetric initial configuration in barycentric coords
    init_cart = np.array([
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
    init_bc = _cartesian_to_barycentric(init_cart)
    init_bc = _project_bary(init_bc)
    init_pts = _barycentric_to_cartesian(init_bc)
    _update_best(init_pts)

    # Phase 1: Global Search with Differential Evolution in barycentric space
    if _elapsed_ok(60):
        def de_obj(flat):
            bc = _project_bary(flat.reshape(-1, 2))
            pts = _barycentric_to_cartesian(bc)
            return -_min_triangle_area(pts)

        bounds = [(0, 1), (0, 1)] * N
        try:
            de_res = differential_evolution(
                de_obj, bounds, maxiter=80, popsize=25,
                seed=123, tol=1e-8, polish=False,
                init='latinhypercube', mutation=(0.5, 1.0), recombination=0.8
            )
            de_bc = _project_bary(de_res.x.reshape(-1, 2))
            de_pts = _barycentric_to_cartesian(de_bc)
            _update_best(de_pts)
        except Exception:
            pass

    # Phase 2: Push-apart local search from best point
    if best_pts is not None and _elapsed_ok(30):
        current_bc = _cartesian_to_barycentric(best_pts)
        current_bc = _project_bary(current_bc)
        current_pts = _barycentric_to_cartesian(current_bc)
        
        # Multiple restarts with different step sizes and perturbations
        for restart in range(8):
            if not _elapsed_ok(5):
                break
            
            if restart > 0:
                # Perturb in barycentric space
                pert = rng.normal(0, 0.02, size=(N, 2))
                current_bc = _project_bary(current_bc + pert)
                current_pts = _barycentric_to_cartesian(current_bc)
            
            # Adaptive step size push-apart
            step = 0.02
            for _ in range(30):
                if not _elapsed_ok(2):
                    break
                
                bc_new, pts_new = _push_apart(current_bc, step=step, n_iter=100)
                area_new = _min_triangle_area(pts_new)
                
                if area_new > _min_triangle_area(current_pts) + 1e-10:
                    current_bc = bc_new
                    current_pts = pts_new
                    _update_best(pts_new)
                else:
                    step *= 0.7
                    if step < 0.001:
                        break

    # Phase 3: Nelder-Mead polish in barycentric space
    if best_pts is not None and _elapsed_ok(10):
        def nm_obj(flat):
            bc = _project_bary(flat.reshape(-1, 2))
            pts = _barycentric_to_cartesian(bc)
            return -_min_triangle_area(pts)
        
        x0 = _cartesian_to_barycentric(best_pts).flatten()
        try:
            res = minimize(
                nm_obj, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'initial_simplex_size': 0.05}
            )
            new_bc = _project_bary(res.x.reshape(-1, 2))
            new_pts = _barycentric_to_cartesian(new_bc)
            _update_best(new_pts)
        except Exception:
            pass

    # Phase 4: Final fine-tune with very small push-apart
    if best_pts is not None and _elapsed_ok(5):
        current_bc = _cartesian_to_barycentric(best_pts)
        current_bc = _project_bary(current_bc)
        current_pts = _barycentric_to_cartesian(current_bc)
        
        for _ in range(50):
            if not _elapsed_ok(1):
                break
            bc_new, pts_new = _push_apart(current_bc, step=0.002, n_iter=200)
            area_new = _min_triangle_area(pts_new)
            if area_new > _min_triangle_area(current_pts) + 1e-12:
                current_bc = bc_new
                current_pts = pts_new
                _update_best(pts_new)
            else:
                break

    if best_pts is None:
        return _barycentric_to_cartesian(_project_bary(init_bc))

    return best_pts
