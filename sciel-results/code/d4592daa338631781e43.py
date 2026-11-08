import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

# Pre-compute all triangle index combinations
_TRIS = list(combinations(range(N), 3))
_TRI_I = np.array([t[0] for t in _TRIS])
_TRI_J = np.array([t[1] for t in _TRIS])
_TRI_K = np.array([t[2] for t in _TRIS])
_N_TRI = len(_TRIS)  # 364

def _get_all_triangle_areas(points):
    """Vectorized computation of all triangle areas."""
    pi = points[_TRI_I]
    pj = points[_TRI_J]
    pk = points[_TRI_K]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return areas

def _min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by 3 points."""
    return np.min(_get_all_triangle_areas(points))

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the score: min triangle area / convex hull area."""
    n = len(points)
    if n < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _normalize_points(points):
    """Center and scale points to unit hull area."""
    points = points - np.mean(points, axis=0)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return points
    scale = 1.0 / np.sqrt(hull_area)
    return points * scale

def _neg_objective(points_flat):
    """Objective for minimization: negative of min triangle area (unit hull area)."""
    points = points_flat.reshape(-1, 2)
    points = _normalize_points(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6
    min_area = _min_triangle_area(points)
    return -min_area

def _smooth_neg_min_area(points_flat, tau=100.0):
    """Smooth surrogate using log-sum-exp."""
    points = points_flat.reshape(-1, 2)
    points = _normalize_points(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6
    areas = _get_all_triangle_areas(points)
    m = np.min(areas)
    shifted = areas - m
    sum_exp = np.sum(np.exp(-tau * shifted))
    smooth_val = -m + (1.0 / tau) * np.log(sum_exp)
    return smooth_val

def _ring_objective(params):
    """
    Objective for two-ring configuration: 7 outer, 7 inner.
    params: [outer_radius, inner_radius, angular_offset]
    Returns negative score (to minimize).
    """
    outer_r = params[0]
    inner_r = params[1]
    offset = params[2]
    
    if outer_r < 0.1 or inner_r < 0.05 or outer_r < inner_r:
        return 1e6
    
    # 7 outer points at radius outer_r
    ang_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    x_outer = outer_r * np.cos(ang_outer)
    y_outer = outer_r * np.sin(ang_outer)
    
    # 7 inner points at radius inner_r, rotated by offset
    ang_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False) + offset
    x_inner = inner_r * np.cos(ang_inner)
    y_inner = inner_r * np.sin(ang_inner)
    
    points = np.column_stack([np.concatenate([x_outer, x_inner]),
                               np.concatenate([y_outer, y_inner])])
    
    # Check all points are on convex hull
    try:
        hull = ConvexHull(points)
        if len(hull.vertices) < N:
            # Soft penalty: count how many are missing
            missing = N - len(hull.vertices)
            penalty = missing * 0.1
        else:
            penalty = 0.0
    except Exception:
        return 1e6
    
    points = _normalize_points(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6
    
    min_area = _min_triangle_area(points)
    score = min_area / hull_area
    return -score + penalty

def _three_ring_objective(params):
    """
    Objective for three-ring configuration: 6 outer, 5 middle, 3 inner.
    params: [r1, r2, r3, offset1, offset2]
    """
    r1, r2, r3 = params[0], params[1], params[2]
    off1, off2 = params[3], params[4]
    
    if r1 < 0.1 or r2 < 0.05 or r3 < 0.02 or r1 < r2 or r2 < r3:
        return 1e6
    
    # 6 outer
    a1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    p1 = np.column_stack([r1*np.cos(a1), r1*np.sin(a1)])
    
    # 5 middle
    a2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + off1
    p2 = np.column_stack([r2*np.cos(a2), r2*np.sin(a2)])
    
    # 3 inner
    a3 = np.linspace(0, 2*np.pi, 3, endpoint=False) + off2
    p3 = np.column_stack([r3*np.cos(a3), r3*np.sin(a3)])
    
    points = np.vstack([p1, p2, p3])
    
    try:
        hull = ConvexHull(points)
        if len(hull.vertices) < N:
            missing = N - len(hull.vertices)
            penalty = missing * 0.1
        else:
            penalty = 0.0
    except Exception:
        return 1e6
    
    points = _normalize_points(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6
    
    min_area = _min_triangle_area(points)
    score = min_area / hull_area
    return -score + penalty

def _parametrized_objective(params):
    """
    Objective for general parametrized optimization.
    params: [dtheta_0..dtheta_13, dr_0..dr_13]
    """
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    dtheta = params[:N]
    dr = params[N:]
    
    angles = base_angles + dtheta
    radii = 1.0 + dr
    
    if np.any(radii <= 0.1):
        return 1e6
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack([x, y])
    
    # Soft convexity penalty
    try:
        hull = ConvexHull(points)
        if len(hull.vertices) < N:
            missing = N - len(hull.vertices)
            penalty = missing * 0.05
        else:
            penalty = 0.0
    except Exception:
        return 1e6
    
    points = _normalize_points(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6
    
    min_area = _min_triangle_area(points)
    score = min_area / hull_area
    return -score + penalty

def _parametrized_smooth_objective(params, tau=100.0):
    """Smooth version of parametrized objective."""
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    dtheta = params[:N]
    dr = params[N:]
    
    angles = base_angles + dtheta
    radii = 1.0 + dr
    
    if np.any(radii <= 0.1):
        return 1e6
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack([x, y])
    
    try:
        hull = ConvexHull(points)
        if len(hull.vertices) < N:
            missing = N - len(hull.vertices)
            penalty = missing * 0.05
        else:
            penalty = 0.0
    except Exception:
        return 1e6
    
    points = _normalize_points(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6
    
    areas = _get_all_triangle_areas(points)
    m = np.min(areas)
    shifted = areas - m
    sum_exp = np.sum(np.exp(-tau * shifted))
    smooth_val = -m + (1.0 / tau) * np.log(sum_exp)
    return smooth_val + penalty

def _local_refine(points, max_iters=200, step_size=0.003):
    """
    Local refinement targeting worst triangles with adaptive steps.
    """
    points = points.copy()
    current_score = _score(points)
    
    for it in range(max_iters):
        areas = _get_all_triangle_areas(points)
        min_area = np.min(areas)
        
        # Find triangles close to minimum
        threshold = min_area + 0.02 * min_area
        min_indices = np.where(areas <= threshold)[0]
        
        # Collect point indices involved in worst triangles
        worst_points = set()
        for tri_idx in min_indices[:25]:
            worst_points.add(_TRI_I[tri_idx])
            worst_points.add(_TRI_J[tri_idx])
            worst_points.add(_TRI_K[tri_idx])
        
        improved = False
        # Try multiple step sizes for each worst point
        for pt_idx in sorted(worst_points):
            for s_mult in [1.0, 0.5, 0.25, 2.0, 4.0]:
                s = step_size * s_mult
                # Try 8 directions
                for dx in [-s, 0, s]:
                    for dy in [-s, 0, s]:
                        if dx == 0 and dy == 0:
                            continue
                        new_points = points.copy()
                        new_points[pt_idx, 0] += dx
                        new_points[pt_idx, 1] += dy
                        new_score = _score(new_points)
                        if new_score > current_score + 1e-12:
                            points = new_points
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try all points with smaller step
            step_size *= 0.7
            if step_size < 1e-10:
                break
            improved = False
            for pt_idx in range(N):
                for dx in [-step_size, step_size]:
                    for dy in [-step_size, step_size]:
                        new_points = points.copy()
                        new_points[pt_idx, 0] += dx
                        new_points[pt_idx, 1] += dy
                        new_score = _score(new_points)
                        if new_score > current_score + 1e-12:
                            points = new_points
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return points, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit
    
    best_score = -1.0
    best_points = None
    
    def _update_best(points):
        nonlocal best_score, best_points
        points = _normalize_points(points)
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    def _elapsed():
        return time.time() - start_time
    
    # --- Phase A: Differential evolution on ring parameters ---
    # Two-ring: 7 outer, 7 inner
    bounds_2ring = [(0.3, 1.5), (0.05, 0.9), (0.0, 2*np.pi/7)]
    try:
        if _elapsed() < deadline - 60:
            res_2ring = differential_evolution(_ring_objective, bounds_2ring,
                                                seed=42, maxiter=300, tol=1e-8,
                                                popsize=15, mutation=(0.3, 1.0),
                                                recombination=0.8, init='latinhypercube')
            if res_2ring.success or res_2ring.fun < 1e5:
                outer_r, inner_r, offset = res_2ring.x
                ang_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
                x_outer = outer_r * np.cos(ang_outer)
                y_outer = outer_r * np.sin(ang_outer)
                ang_inner = np.linspace(0, 2*np.pi, 7, endpoint=False) + offset
                x_inner = inner_r * np.cos(ang_inner)
                y_inner = inner_r * np.sin(ang_inner)
                points = np.column_stack([np.concatenate([x_outer, x_inner]),
                                           np.concatenate([y_outer, y_inner])])
                _update_best(points)
    except Exception:
        pass
    
    # Three-ring: 6+5+3
    bounds_3ring = [(0.3, 1.5), (0.1, 1.2), (0.02, 0.8), (0.0, 2*np.pi/5), (0.0, 2*np.pi/3)]
    try:
        if _elapsed() < deadline - 45:
            res_3ring = differential_evolution(_three_ring_objective, bounds_3ring,
                                                seed=43, maxiter=250, tol=1e-8,
                                                popsize=15, mutation=(0.3, 1.0),
                                                recombination=0.8, init='latinhypercube')
            if res_3ring.success or res_3ring.fun < 1e5:
                r1, r2, r3, off1, off2 = res_3ring.x
                a1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
                p1 = np.column_stack([r1*np.cos(a1), r1*np.sin(a1)])
                a2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + off1
                p2 = np.column_stack([r2*np.cos(a2), r2*np.sin(a2)])
                a3 = np.linspace(0, 2*np.pi, 3, endpoint=False) + off2
                p3 = np.column_stack([r3*np.cos(a3), r3*np.sin(a3)])
                points = np.vstack([p1, p2, p3])
                _update_best(points)
    except Exception:
        pass
    
    # --- Phase B: General parametrized DE (14-gon with perturbations) ---
    try:
        if _elapsed() < deadline - 30:
            bounds_param = [(-0.3, 0.3)] * N + [(0.2, 2.0)] * N
            res_param = differential_evolution(_parametrized_objective, bounds_param,
                                                seed=44, maxiter=100, tol=1e-6,
                                                popsize=12, mutation=(0.3, 1.0),
                                                recombination=0.8, init='latinhypercube')
            if res_param.fun < 1e5:
                base_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
                dtheta = res_param.x[:N]
                dr = res_param.x[N:]
                angles = base_angles + dtheta
                radii = dr
                x = radii * np.cos(angles)
                y = radii * np.sin(angles)
                points = np.column_stack([x, y])
                _update_best(points)
    except Exception:
        pass
    
    # --- Phase C: Cartesian optimization from multiple starts ---
    configs = []
    
    # Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # From ring solutions if available
    if best_points is not None:
        configs.append(best_points.copy())
        # Perturbed versions
        for _ in range(3):
            configs.append(best_points + 0.02 * np.random.randn(N, 2))
    
    # Two-ring with optimized params
    outer_r, inner_r, offset = 1.0, 0.5, np.pi/7
    ang_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    ang_inner = np.linspace(0, 2*np.pi, 7, endpoint=False) + offset
    configs.append(np.column_stack([np.concatenate([outer_r*np.cos(ang_outer), inner_r*np.cos(ang_inner)]),
                                     np.concatenate([outer_r*np.sin(ang_outer), inner_r*np.sin(ang_inner)])]))
    
    # Random starts
    for _ in range(4):
        configs.append(np.random.randn(N, 2))
    
    tau_values = [50.0, 150.0]
    
    for x_init in configs:
        if _elapsed() > deadline - 15:
            break
        
        for tau in tau_values:
            if _elapsed() > deadline - 12:
                break
            
            try:
                res_smooth = minimize(lambda x, t=tau: _smooth_neg_min_area(x, tau=t),
                                      x_init.flatten(),
                                      method='Powell',
                                      options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-10, 'adaptive': True})
                
                if _elapsed() > deadline - 10:
                    break
                    
                p_opt = res_smooth.x.reshape(-1, 2)
                p_opt = _normalize_points(p_opt)
                
                # Polish with exact objective
                res_exact = minimize(_neg_objective,
                                     p_opt.flatten(),
                                     method='Nelder-Mead',
                                     options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
                
                if _elapsed() > deadline - 8:
                    break
                    
                p_polished = res_exact.x.reshape(-1, 2)
                p_polished = _normalize_points(p_polished)
                
                _update_best(p_polished)
                
            except Exception:
                pass
    
    # --- Phase D: Local refinement ---
    if best_points is not None and _elapsed() < deadline - 5:
        refined, refined_score = _local_refine(best_points, max_iters=200, step_size=0.003)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined
    
    # --- Phase E: Final polish ---
    if best_points is not None and _elapsed() < deadline - 2:
        try:
            res_final = minimize(_neg_objective,
                                 best_points.flatten(),
                                 method='Nelder-Mead',
                                 options={'maxiter': 1500, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            p_final = res_final.x.reshape(-1, 2)
            p_final = _normalize_points(p_final)
            _update_best(p_final)
        except Exception:
            pass
    
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_points = _normalize_points(best_points)
    
    return best_points
