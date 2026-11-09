import numpy as np
import time
import itertools
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
_DEADLINE = 142.0  # Safety margin before 150s kill

def _params_to_points(params):
    """Convert angular/radial perturbation params to 2D points.
    params shape (N, 2): [angle_offset, radial_scale] for each vertex of regular N-gon.
    Ensures strict convexity by construction."""
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angle_offsets = params[:, 0]
    radial_scales = params[:, 1]
    # Constrain angles to maintain ordering (convexity)
    # Small angular perturbations around base angles
    angles = base_angles + angle_offsets
    # Sort by angle to maintain convex polygon order
    order = np.argsort(angles)
    angles_sorted = angles[order]
    radials_sorted = radial_scales[order]
    # Ensure positive radii
    radials_sorted = np.maximum(radials_sorted, 0.1)
    x = radials_sorted * np.cos(angles_sorted)
    y = radials_sorted * np.sin(angles_sorted)
    return np.column_stack([x, y])

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas."""
    n = len(pts)
    idx = list(itertools.combinations(range(n), 3))
    i_idx = np.array([c[0] for c in idx])
    j_idx = np.array([c[1] for c in idx])
    k_idx = np.array([c[2] for c in idx])
    v1 = pts[j_idx] - pts[i_idx]
    v2 = pts[k_idx] - pts[i_idx]
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return areas

def _soft_min(areas, tau):
    """Soft minimum via log-sum-exp."""
    m = np.min(areas)
    shifted = areas - m
    shifted = np.clip(shifted, 0, 50.0 / tau)  # numerical stability
    sum_exp = np.sum(np.exp(-tau * shifted))
    return m - (1.0 / tau) * np.log(sum_exp)

def _min_triangle_area(pts):
    """Minimum triangle area."""
    return np.min(_triangle_areas(pts))

def _hull_area(pts):
    """Convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 0.0
    ma = _min_triangle_area(pts)
    return ma / ha

def _normalize(pts):
    """Center and scale to unit hull area."""
    pts = pts - np.mean(pts, axis=0)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return pts
    return pts / np.sqrt(ha)

def _neg_objective(flat):
    """Negative score after normalization."""
    pts = flat.reshape(-1, 2)
    pts = _normalize(pts)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e6
    return -_min_triangle_area(pts)

def _smooth_objective(flat, tau=50.0):
    """Smooth surrogate: minimize -soft_min(areas)/hull_area."""
    pts = flat.reshape(-1, 2)
    pts = _normalize(pts)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e6
    areas = _triangle_areas(pts)
    sm = _soft_min(areas, tau)
    return -sm / ha

def _neg_objective_param(params):
    """Negative score using angular/radial parametrization."""
    pts = _params_to_points(params.reshape(-1, 2))
    pts = _normalize(pts)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e6
    return -_min_triangle_area(pts)

def _smooth_objective_param(params, tau=50.0):
    """Smooth surrogate using angular/radial parametrization."""
    pts = _params_to_points(params.reshape(-1, 2))
    pts = _normalize(pts)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e6
    areas = _triangle_areas(pts)
    sm = _soft_min(areas, tau)
    return -sm / ha

def _points_to_params(pts):
    """Convert points to angular/radial params relative to regular N-gon."""
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    pts_c = pts - np.array([cx, cy])
    angles = np.arctan2(pts_c[:, 1], pts_c[:, 0])
    radii = np.sqrt(pts_c[:, 0]**2 + pts_c[:, 1]**2)
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Unwrap angle differences
    angle_diffs = np.unwrap(angles - base_angles)
    # Scale radii relative to mean
    mean_r = np.mean(radii)
    if mean_r < 1e-10:
        return np.zeros((N, 2))
    radial_scales = radii / mean_r
    return np.column_stack([angle_diffs, radial_scales])

def _local_refine(pts, max_iters=300, base_step=0.01, deadline=None):
    """Fast local refinement: perturb points in worst triangles using in-place updates."""
    pts = pts.copy()
    current = _score(pts)
    n = len(pts)
    idx = list(itertools.combinations(range(n), 3))
    idx_arr = np.array(idx)  # shape (C(n,3), 3)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        areas = _triangle_areas(pts)
        min_a = np.min(areas)
        # Find points in triangles with area close to minimum
        tol = 1e-6
        worst_mask = np.abs(areas - min_a) < tol
        worst_tris = np.where(worst_mask)[0][:30]
        worst_pts = set()
        for ti in worst_tris:
            for p in idx_arr[ti]:
                worst_pts.add(p)
        
        if not worst_pts:
            break
        
        improved = False
        step = base_step * (0.7 ** (it // 30))
        for pi in sorted(worst_pts):
            # Try 8 directions + radial
            for dx, dy in [(-step, 0), (step, 0), (0, -step), (0, step),
                          (-step, -step), (step, step), (-step, step), (step, -step)]:
                orig_x, orig_y = pts[pi, 0], pts[pi, 1]
                pts[pi, 0] += dx
                pts[pi, 1] += dy
                s = _score(pts)
                if s > current + 1e-12:
                    current = s
                    improved = True
                    break
                else:
                    pts[pi, 0] = orig_x
                    pts[pi, 1] = orig_y
            if improved:
                break
        if not improved:
            # Try smaller steps
            if step < 1e-6:
                break
            base_step *= 0.5
    return pts, current

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_score = -1.0
    best_points = None
    
    # Generate diverse initial configurations in parameter space
    param_configs = []
    
    # 1. Regular 14-gon (zero perturbations)
    param_configs.append(np.zeros((N, 2)))
    
    # 2. Small random angular/radial perturbations
    for scale_a, scale_r in [(0.02, 0.02), (0.05, 0.05), (0.1, 0.1), (0.03, 0.08)]:
        param_configs.append(np.column_stack([
            scale_a * np.random.randn(N),
            1.0 + scale_r * np.random.randn(N)
        ]))
    
    # 3. Structured radial perturbations (alternating in/out)
    for amp in [0.05, 0.1, 0.15]:
        radial = 1.0 + amp * np.array([1, -1] * (N // 2) + [1] if N % 2 else [1, -1] * (N // 2))
        param_configs.append(np.column_stack([np.zeros(N), radial]))
    
    # 4. Rotational symmetry breaking
    for phase in [np.pi / 14, 2 * np.pi / 14, np.pi / 7]:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + phase
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        param_configs.append(_points_to_params(pts))
    
    # 5. Random parameters
    for _ in range(4):
        param_configs.append(np.column_stack([
            0.05 * np.random.randn(N),
            1.0 + 0.1 * np.random.randn(N)
        ]))
    
    # Also keep some raw point configs for diversity
    point_configs = []
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    point_configs.append(x0.copy())
    for scale in [0.03, 0.05, 0.1]:
        point_configs.append(x0 + scale * np.random.randn(N, 2))
    
    # Optimize in parameter space first
    for params_init in param_configs:
        if time.time() > deadline - 25:
            break
        
        try:
            p_flat = params_init.flatten()
            
            # Phase 1: Smooth optimization in param space with Powell
            res1 = minimize(_smooth_objective_param, p_flat, args=(100.0,),
                          method='Powell',
                          options={'maxiter': 150, 'xtol': 1e-8, 'ftol': 1e-10, 'adaptive': True})
            
            if time.time() > deadline - 20:
                break
            
            p1_params = res1.x.reshape(-1, 2)
            p1_pts = _params_to_points(p1_params)
            p1_pts = _normalize(p1_pts)
            
            # Phase 2: Refine in Cartesian space with Nelder-Mead
            res2 = minimize(_neg_objective, p1_pts.flatten(),
                          method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            if time.time() > deadline - 15:
                break
            
            p2 = res2.x.reshape(-1, 2)
            p2 = _normalize(p2)
            
            # Phase 3: Local refinement
            if time.time() < deadline - 10:
                p3, s3 = _local_refine(p2, max_iters=150, base_step=0.005, deadline=deadline - 7)
            else:
                p3, s3 = p2, _score(p2)
            
            s = _score(p3)
            if s > best_score:
                best_score = s
                best_points = p3.copy()
                
        except Exception:
            pass
    
    # Optimize some raw point configs
    for x_init in point_configs:
        if time.time() > deadline - 15:
            break
        
        try:
            x_flat = x_init.flatten()
            
            res1 = minimize(_smooth_objective, x_flat, args=(80.0,),
                          method='Powell',
                          options={'maxiter': 100, 'xtol': 1e-8, 'ftol': 1e-10, 'adaptive': True})
            
            if time.time() > deadline - 12:
                break
            
            p1 = res1.x.reshape(-1, 2)
            p1 = _normalize(p1)
            
            res2 = minimize(_neg_objective, p1.flatten(),
                          method='Nelder-Mead',
                          options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            if time.time() > deadline - 10:
                break
            
            p2 = res2.x.reshape(-1, 2)
            p2 = _normalize(p2)
            
            if time.time() < deadline - 8:
                p3, s3 = _local_refine(p2, max_iters=100, base_step=0.005, deadline=deadline - 6)
            else:
                p3, s3 = p2, _score(p2)
            
            s = _score(p3)
            if s > best_score:
                best_score = s
                best_points = p3.copy()
                
        except Exception:
            pass
    
    if best_points is None:
        best_points = _normalize(x0.copy())
    
    # Final high-precision polish in Cartesian space
    if best_points is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(_neg_objective, best_points.flatten(),
                              method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            pf = res_final.x.reshape(-1, 2)
            pf = _normalize(pf)
            sf = _score(pf)
            if sf > best_score:
                best_points = pf
                best_score = sf
        except Exception:
            pass
    
    # Second polish pass if time allows
    if best_points is not None and time.time() < deadline - 3:
        try:
            p_final = best_points.copy()
            p_final, s_final = _local_refine(p_final, max_iters=200, base_step=0.001, deadline=deadline - 1)
            if s_final > best_score:
                best_points = p_final
                best_score = s_final
        except Exception:
            pass
    
    return best_points
