"""Heilbronn-11: Topological Active-Set Pivot with Delaunay-aware refinement."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE = 142.0

# Precompute all triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
N_TRIP = len(_TRIPLETS)  # 165

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])


def _triangle_areas(pts):
    """Compute all 165 triangle areas from N points."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return np.abs(cross) * 0.5


def _min_area(pts):
    """Return minimum triangle area."""
    return np.min(_triangle_areas(pts))


def _area_gradients(pts):
    """Compute gradient of each triangle area w.r.t. point coordinates.
    Returns array of shape (N_TRIP, N, 2) where grad[k, i, :] is d(area_k)/d(pts[i]).
    
    For triangle with vertices A, B, C:
    area = 0.5 * |cross(B-A, C-A)|
    d(area)/dA = -0.5 * perp(C-B)  (where perp(x,y) = (-y, x))
    d(area)/dB =  0.5 * perp(C-A)
    d(area)/dC = -0.5 * perp(B-A)
    """
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    
    # Signed cross product for gradient direction
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2  # signed area * 2
    
    # Sign of each triangle's orientation
    sign = np.sign(cross)
    sign[sign == 0] = 1.0  # avoid zero
    
    # Gradient contributions: for triangle (i,j,k) with signed area s = cross/2
    # d(s)/dpi = -0.5 * perp(pj - pk)
    # d(s)/dpj =  0.5 * perp(pk - pi)  
    # d(s)/dpk = -0.5 * perp(pi - pj)
    # For absolute area, multiply by sign(cross)
    
    grad = np.zeros((N_TRIP, N, 2))
    
    # perp(v) = (-vy, vx)
    # d(cross)/dpi = (-(dy2-dy1), dx2-dx1) = (dy1-dy2, dx2-dx1)
    # Actually let me be more careful:
    # cross = (xj-xi)*(yk-yi) - (yj-yi)*(xk-xi)
    # d(cross)/dxi = -(yk-yi) + (yj-yi) = yj - yk
    # d(cross)/dyi = (xk-xi) - (xj-xi) = xk - xj
    # d(cross)/dxj = (yk-yi)
    # d(cross)/dyj = -(xk-xi)
    # d(cross)/dxk = -(yj-yi)
    # d(cross)/dyk = (xj-xi)
    
    # area = 0.5 * |cross|, so d(area)/d = 0.5 * sign(cross) * d(cross)/d
    
    factor = 0.5 * sign
    
    # For point i (first vertex)
    grad[:, _T_I, 0] += factor * (dy1 - dy2)  # d(cross)/dxi = yj-yk... wait
    # Let me recompute:
    # cross = (xj-xi)(yk-yi) - (yj-yi)(xk-xi)
    # d/dxi: -(yk-yi) + (yj-yi) = yj - yk
    # d/dyi: (xk-xi) - (xj-xi) = xk - xj
    # d/dxj: (yk-yi)
    # d/dyj: -(xk-xi)
    # d/dxk: -(yj-yi)
    # d/dyk: (xj-xi)
    
    grad[:, _T_I, 0] = factor * (dy1 - dy2)  # yj - yk
    grad[:, _T_I, 1] = factor * (dx2 - dx1)  # xk - xj
    
    grad[:, _T_J, 0] = factor * dy2  # yk - yi
    grad[:, _T_J, 1] = factor * (-dx2)  # -(xk - xi)
    
    grad[:, _T_K, 0] = factor * (-dy1)  # -(yj - yi)
    grad[:, _T_K, 1] = factor * dx1  # xj - xi
    
    return grad


def _project_cartesian(pts):
    """Project points into the triangle."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _is_valid_cart(pts, tol=1e-6):
    """Check if all points are inside the triangle."""
    for i in range(len(pts)):
        x, y = pts[i]
        if y < -tol or y > S3H + tol:
            return False
        x_min = y / S3
        x_max = 1 - y / S3
        if x < x_min - tol or x > x_max + tol:
            return False
    return True


def _active_set_pivot(pts, max_iter=50, tol=1e-9):
    """Topological Active-Set Pivot:
    1. Find the minimum-area triangle (critical triangle)
    2. Compute gradients of all triangle areas
    3. Solve for a direction that increases the min area while keeping others >= min
    4. Line search along this direction
    Repeat until convergence or max_iter.
    """
    pts = pts.copy()
    current_min = _min_area(pts)
    
    for it in range(max_iter):
        areas = _triangle_areas(pts)
        min_idx = np.argmin(areas)
        min_val = areas[min_idx]
        
        if min_val - current_min < tol:
            break
        current_min = min_val
        
        # Get gradients
        grads = _area_gradients(pts)  # shape (N_TRIP, N, 2)
        
        # Active set: triangles with area close to min
        active_mask = areas < min_val + 0.01 * min_val + 1e-8
        active_grads = grads[active_mask]  # (K, N, 2)
        K = active_grads.shape[0]
        
        # We want to find direction d (N*2,) that maximizes increase in min area
        # subject to: for each active triangle k, d(area_k)/dt >= 0
        # and for inactive triangles, d(area_k)/dt >= -small (allow slight decrease)
        
        # Simplified: use the average gradient of active triangles as direction
        # This is a subgradient method for the max-min problem
        
        # The subgradient of min(areas) is the average of gradients of min-achieving triangles
        if K > 0:
            # Average gradient over active set
            avg_grad = np.mean(active_grads, axis=0)  # (N, 2)
            
            # Also consider: we want to move points to increase separation
            # Add a small repulsion term to avoid degenerate configurations
            # Compute pairwise distances and add repulsive forces
            repulsion = np.zeros((N, 2))
            for i in range(N):
                for j in range(i + 1, N):
                    d = pts[i] - pts[j]
                    dist = np.linalg.norm(d)
                    if dist < 0.05:
                        # Strong repulsion for close points
                        repulsion[i] += d / (dist + 1e-10) * 0.1
                        repulsion[j] -= d / (dist + 1e-10) * 0.1
                    elif dist < 0.15:
                        repulsion[i] += d / (dist + 1e-10) * 0.01
                        repulsion[j] -= d / (dist + 1e-10) * 0.01
            
            direction = avg_grad.flatten() + repulsion.flatten()
            
            # Normalize direction
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-12:
                break
            direction = direction / dir_norm
            
            # Line search: try several step sizes
            best_step = 0.0
            best_area = min_val
            for alpha in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]:
                new_pts = pts + alpha * direction.reshape(N, 2)
                new_pts = _project_cartesian(new_pts)
                if not _is_valid_cart(new_pts):
                    continue
                new_min = _min_area(new_pts)
                if new_min > best_area:
                    best_area = new_min
                    best_step = alpha
            
            if best_step > 0:
                pts = pts + best_step * direction.reshape(N, 2)
                pts = _project_cartesian(pts)
            else:
                # Try smaller steps
                for alpha in [0.001, 0.002, 0.005]:
                    new_pts = pts + alpha * direction.reshape(N, 2)
                    new_pts = _project_cartesian(new_pts)
                    if _is_valid_cart(new_pts):
                        new_min = _min_area(new_pts)
                        if new_min > min_val:
                            pts = new_pts
                            break
                else:
                    break
    
    return pts


def _delaunay_refine(pts, max_iter=30):
    """Use Delaunay triangulation to identify critical triangles and refine.
    The Delaunay triangulation gives a natural partition; the minimum area
    triangle in the Delaunay is often critical.
    """
    pts = pts.copy()
    
    for it in range(max_iter):
        try:
            tri = Delaunay(pts)
            # Get triangle indices
            tri_idx = tri.simplices  # (M, 3)
            
            # Compute areas of Delaunay triangles
            tri_areas = []
            for t in tri_idx:
                a = 0.5 * abs(np.cross(pts[t[1]] - pts[t[0]], pts[t[2]] - pts[t[0]]))
                tri_areas.append(a)
            tri_areas = np.array(tri_areas)
            
            # Find minimum area Delaunay triangle
            min_tri_idx = np.argmin(tri_areas)
            min_tri = tri_idx[min_tri_idx]
            min_area_val = tri_areas[min_tri_idx]
            
            # Also check all triangles, not just Delaunay
            all_areas = _triangle_areas(pts)
            global_min = np.min(all_areas)
            
            # If the global min is not in the Delaunay, we still need to handle it
            # For now, focus on Delaunay triangles for structural insight
            
            # Perturb the vertices of the minimum triangle outward
            # to increase its area while checking constraints
            i, j, k = min_tri
            
            # Compute centroid of the triangle
            centroid = (pts[i] + pts[j] + pts[k]) / 3.0
            
            # Try to expand the triangle by moving vertices away from centroid
            improved = False
            for alpha in [0.02, 0.01, 0.005]:
                new_pts = pts.copy()
                for idx in [i, j, k]:
                    v = pts[idx] - centroid
                    v_norm = np.linalg.norm(v)
                    if v_norm > 1e-10:
                        v = v / v_norm
                        new_pts[idx] = pts[idx] + alpha * v
                
                new_pts = _project_cartesian(new_pts)
                if not _is_valid_cart(new_pts):
                    continue
                
                new_min = _min_area(new_pts)
                if new_min > global_min:
                    pts = new_pts
                    improved = True
                    break
            
            if not improved:
                break
                
        except Exception:
            break
    
    return pts


def _soft_min_objective(pts_flat, beta):
    """Soft minimum objective for L-BFGS-B."""
    pts = pts_flat.reshape(N, 2)
    pts = _project_cartesian(pts)
    areas = _triangle_areas(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    # Clip for numerical stability
    shifted = np.clip(shifted, -100, 100)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return -soft_min


def _soft_min_grad(pts_flat, beta):
    """Gradient of soft minimum objective."""
    pts = pts_flat.reshape(N, 2)
    areas = _triangle_areas(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    shifted = np.clip(shifted, -100, 100)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp  # softmax weights
    
    # Gradient of -soft_min = gradient of logsumexp(beta*(areas - max_a)) / beta
    # = (1/beta) * sum_k weights_k * beta * d(area_k)/d(pts)
    # = sum_k weights_k * d(area_k)/d(pts)
    
    grads = _area_gradients(pts)  # (N_TRIP, N, 2)
    grad = np.sum(weights[:, np.newaxis, np.newaxis] * grads, axis=0)  # (N, 2)
    
    return -grad.flatten()


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(pts):
        nonlocal best_pts, best_area
        if not _is_valid_cart(pts):
            return
        area = _min_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
    
    # Initial configuration
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
    
    def _start_from_base(base, scale):
        pts = base + rng.normal(0, scale, size=(N, 2))
        pts = _project_cartesian(pts)
        return pts
    
    # Phase A: Multi-start with Active-Set Pivot
    n_starts = 25
    bases = [init]
    for _ in range(12):
        bases.append(_start_from_base(init, 0.2))
    
    for s in range(n_starts):
        if not _elapsed_ok(50):
            break
        
        base = bases[s % len(bases)]
        pts = _start_from_base(base, 0.05)
        _update_best(pts)
        
        # Apply Active-Set Pivot
        pts = _active_set_pivot(pts, max_iter=40)
        _update_best(pts)
        
        # Apply Delaunay refinement
        if _elapsed_ok(50):
            pts = _delaunay_refine(pts, max_iter=20)
            _update_best(pts)
    
    # Phase B: Soft-min L-BFGS-B polish on best
    if best_pts is not None and _elapsed_ok(20):
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if not _elapsed_ok(15):
                break
            try:
                result = minimize(
                    _soft_min_objective,
                    best_pts.flatten(),
                    jac=_soft_min_grad,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                pts = result.x.reshape(N, 2)
                pts = _project_cartesian(pts)
                _update_best(pts)
            except Exception:
                pass
    
    # Phase C: Final Active-Set Pivot polish
    if best_pts is not None and _elapsed_ok(10):
        pts = _active_set_pivot(best_pts, max_iter=60)
        _update_best(pts)
    
    # Phase D: Nelder-Mead final polish
    if best_pts is not None and _elapsed_ok(5):
        def nm_obj(x):
            p = x.reshape(N, 2)
            p = _project_cartesian(p)
            return -_min_area(p)
        
        try:
            res = minimize(nm_obj, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12})
            p = res.x.reshape(N, 2)
            p = _project_cartesian(p)
            _update_best(p)
        except Exception:
            pass
    
    # Fallback
    if best_pts is None:
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = v + 0.5 * (1 - u - v)
            y = (1 - u - v) * S3H
            pts[i] = [x, y]
        return pts
    
    best_pts = _project_cartesian(best_pts)
    return best_pts

# EVOLVE-BLOCK-END
