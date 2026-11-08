import numpy as np
import time
from scipy.optimize import minimize
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

def _local_refine(points, max_iters=100, step_size=0.005):
    """
    Local refinement that targets the worst triangles.
    Uses a more aggressive search with adaptive step sizes.
    """
    points = points.copy()
    current_score = _score(points)
    
    for it in range(max_iters):
        areas = _get_all_triangle_areas(points)
        min_area = np.min(areas)
        
        # Find triangles close to minimum
        threshold = min_area + 0.01 * min_area
        min_indices = np.where(areas <= threshold)[0]
        
        # Collect point indices involved in worst triangles
        worst_points = set()
        for tri_idx in min_indices[:20]:
            worst_points.add(_TRI_I[tri_idx])
            worst_points.add(_TRI_J[tri_idx])
            worst_points.add(_TRI_K[tri_idx])
        
        improved = False
        # Try multiple step sizes for each worst point
        for pt_idx in list(worst_points):
            for s_mult in [1.0, 0.5, 0.25, 2.0]:
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
            if step_size < 1e-8:
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

def _parametrized_objective(params):
    """
    Objective for parametrized optimization.
    params: [dtheta_0..dtheta_13, dr_0..dr_13]
    dtheta: angular perturbations (radians)
    dr: radial perturbations
    """
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    dtheta = params[:N]
    dr = params[N:]
    
    angles = base_angles + dtheta
    radii = 1.0 + dr
    
    # Ensure radii are positive
    if np.any(radii <= 0.1):
        return 1e6
    
    # Check for valid convex configuration
    # Points must be in convex position
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack([x, y])
    
    # Check if all points are on the convex hull (strictly convex)
    try:
        hull = ConvexHull(points)
        if len(hull.vertices) < N:
            # Not all points are on the hull - penalty
            return 1e6
    except Exception:
        return 1e6
    
    points = _normalize_points(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6
    
    min_area = _min_triangle_area(points)
    return -min_area

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
            return 1e6
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
    return smooth_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit
    
    best_score = -1.0
    best_points = None
    
    def _update_best(points):
        nonlocal best_score, best_points
        points = _normalize_points(points)
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    # Generate initial configurations
    configs = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(x0_reg)
    
    # 2. Regular 14-gon with noise
    configs.append(x0_reg + 0.03 * np.random.randn(N, 2))
    
    # 3. Two rings: 7 outer, 7 inner
    ang1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer = np.column_stack([np.cos(ang1), np.sin(ang1)])
    ang2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi/7
    inner = 0.5 * np.column_stack([np.cos(ang2), np.sin(ang2)])
    configs.append(np.vstack([outer, inner]))
    
    # 4. Three rings: 5, 5, 4
    ang1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    ring1 = np.column_stack([np.cos(ang1), np.sin(ang1)])
    ang2 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi/5
    ring2 = 0.6 * np.column_stack([np.cos(ang2), np.sin(ang2)])
    ang3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/4
    ring3 = 0.3 * np.column_stack([np.cos(ang3), np.sin(ang3)])
    configs.append(np.vstack([ring1, ring2, ring3]))
    
    # 5. Fibonacci spiral
    golden = (1 + np.sqrt(5)) / 2
    fib_angles = np.arange(N) * 2 * np.pi / (golden * golden)
    fib_radii = np.sqrt(np.arange(N) / N)
    x0_fib = np.column_stack([fib_radii * np.cos(fib_angles), fib_radii * np.sin(fib_angles)])
    configs.append(x0_fib)
    
    # 6. Random points
    configs.append(np.random.randn(N, 2))
    
    # 7. More diverse random starts
    for _ in range(5):
        configs.append(np.random.randn(N, 2))
    
    # --- Phase A: Cartesian optimization for all configs ---
    tau_values = [50.0, 100.0, 200.0]
    
    for x_init in configs:
        if time.time() > deadline - 35:
            break
        
        for tau in tau_values:
            if time.time() > deadline - 30:
                break
            
            try:
                # Smooth optimization with Powell
                res_smooth = minimize(lambda x, t=tau: _smooth_neg_min_area(x, tau=t),
                                      x_init.flatten(),
                                      method='Powell',
                                      options={'maxiter': 300, 'xtol': 1e-8, 'ftol': 1e-10, 'adaptive': True})
                
                if time.time() > deadline - 25:
                    break
                    
                p_opt = res_smooth.x.reshape(-1, 2)
                p_opt = _normalize_points(p_opt)
                
                # Polish with exact objective using Nelder-Mead
                res_exact = minimize(_neg_objective,
                                     p_opt.flatten(),
                                     method='Nelder-Mead',
                                     options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
                
                if time.time() > deadline - 20:
                    break
                    
                p_polished = res_exact.x.reshape(-1, 2)
                p_polished = _normalize_points(p_polished)
                
                _update_best(p_polished)
                
            except Exception:
                pass
    
    # --- Phase B: Parametrized optimization (regular polygon with perturbations) ---
    # Start from zero perturbations (regular polygon)
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Multiple parametrized starts
    param_starts = [np.zeros(2 * N)]
    # Add some random perturbations
    for _ in range(8):
        param_starts.append(0.05 * np.random.randn(2 * N))
    
    for p_init in param_starts:
        if time.time() > deadline - 15:
            break
        
        try:
            # Smooth optimization in parameter space
            res_param_smooth = minimize(lambda x: _parametrized_smooth_objective(x, tau=100.0),
                                        p_init,
                                        method='Powell',
                                        options={'maxiter': 250, 'xtol': 1e-8, 'ftol': 1e-10, 'adaptive': True})
            
            if time.time() > deadline - 10:
                break
            
            p_opt_params = res_param_smooth.x
            base_angles_local = np.linspace(0, 2 * np.pi, N, endpoint=False)
            dtheta = p_opt_params[:N]
            dr = p_opt_params[N:]
            angles = base_angles_local + dtheta
            radii = 1.0 + dr
            x = radii * np.cos(angles)
            y = radii * np.sin(angles)
            p_opt = np.column_stack([x, y])
            p_opt = _normalize_points(p_opt)
            
            # Polish in Cartesian space
            res_exact = minimize(_neg_objective,
                                 p_opt.flatten(),
                                 method='Nelder-Mead',
                                 options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            
            if time.time() > deadline - 5:
                break
            
            p_polished = res_exact.x.reshape(-1, 2)
            p_polished = _normalize_points(p_polished)
            
            _update_best(p_polished)
            
        except Exception:
            pass
    
    # --- Phase C: Local refinement ---
    if best_points is not None and time.time() < deadline - 3:
        refined, refined_score = _local_refine(best_points, max_iters=150, step_size=0.005)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined
    
    # --- Phase D: Additional polish with very tight tolerances ---
    if best_points is not None and time.time() < deadline - 1:
        try:
            res_final = minimize(_neg_objective,
                                 best_points.flatten(),
                                 method='Nelder-Mead',
                                 options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            p_final = res_final.x.reshape(-1, 2)
            p_final = _normalize_points(p_final)
            _update_best(p_final)
        except Exception:
            pass
    
    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_points = _normalize_points(best_points)
    
    return best_points
