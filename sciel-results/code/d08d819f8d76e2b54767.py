"""
Bootstrap spawn: MODEL-AND-OPTIMIZE approach.

Parameterize the 14 points in a structured way (e.g., radii and angles for a
convex-positioned set, or a two-ring model) and fit the parameters directly
against the task's objective: maximize min_triangle_area / hull_area.

The key insight for "maximize the minimum triangle area" with N points in a
convex region is that an optimal configuration tends to be close to a regular
polygon or a perturbation thereof. We use a parameterized model:

  - Outer ring: K points on an ellipse or perturbed circle (these define the hull)
  - Inner ring: (N-K) points at smaller radii

Then we optimize the parameters (radii, angles, possibly elliptic eccentricity)
directly.

For N=14, a good structure is: 10 points on an outer ring (roughly decagonal)
and 4 points on an inner ring. This balances the minimum triangle area.

We use scipy.optimize with a smooth surrogate (soft-min of triangle areas)
combined with the exact score for final evaluation.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
DEADLINE = 143.0  # Safety margin before 150s kill


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            local_min = 0.5 * np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area


def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3 = 364)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)


def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_a


def _neg_score(pts_flat):
    """Negative score for minimization."""
    pts = pts_flat.reshape(-1, 2)
    s = _score(pts)
    return -s


def _soft_obj(pts_flat, beta=100.0):
    """Soft objective: hull_area / soft_min(triangle areas)."""
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val


def _soft_obj_grad(pts_flat, beta=100.0):
    """Soft objective with numerical gradient (for L-BFGS-B)."""
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10, None
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10, None
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10, None
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10, None
    return hull_area / soft_min_val, None


def _two_ring_params_to_pts(outer_n, inner_n, params):
    """
    Convert two-ring model parameters to point coordinates.
    
    params layout:
      [0:outer_n]   outer angles (radians)
      [outer_n:outer_n+inner_n] inner angles (radians)
      [outer_n+inner_n]         outer radius scale
      [outer_n+inner_n+1]       inner radius scale
      [outer_n+inner_n+2]       elliptic x-scale
      [outer_n+inner_n+3]       elliptic y-scale
      [outer_n+inner_n+4]       outer angular perturbation amplitude
      [outer_n+inner_n+5]       inner angular perturbation amplitude
    """
    outer_angles = params[0:outer_n]
    inner_angles = params[outer_n:outer_n+inner_n]
    outer_r = params[outer_n+inner_n]
    inner_r = params[outer_n+inner_n+1]
    ex = params[outer_n+inner_n+2]
    ey = params[outer_n+inner_n+3]
    outer_pert = params[outer_n+inner_n+4]
    inner_pert = params[outer_n+inner_n+5]
    
    n_total = outer_n + inner_n
    base_outer_angles = 2 * np.pi * np.arange(outer_n) / outer_n
    base_inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    
    outer_angles_actual = base_outer_angles + outer_pert * np.sin(2 * base_outer_angles) + outer_angles
    inner_angles_actual = base_inner_angles + inner_pert * np.sin(3 * base_inner_angles) + inner_angles
    
    outer_pts = np.column_stack([
        outer_r * ex * np.cos(outer_angles_actual),
        outer_r * ey * np.sin(outer_angles_actual)
    ])
    inner_pts = np.column_stack([
        inner_r * ex * np.cos(inner_angles_actual),
        inner_r * ey * np.sin(inner_angles_actual)
    ])
    
    pts = np.vstack([outer_pts, inner_pts])
    return pts


def _optimize_two_ring(time_budget):
    """
    Optimize a two-ring model: 10 outer + 4 inner points.
    Parameters are angles, radii, elliptic scales, and perturbation amplitudes.
    """
    start_time = time.time()
    
    outer_n = 10
    inner_n = 4
    
    def obj_func(params):
        if time.time() - start_time > time_budget:
            raise TimeoutError()
        pts = _two_ring_params_to_pts(outer_n, inner_n, params)
        val, grad = _soft_obj_grad(pts.flatten(), beta=100.0)
        return val, grad
    
    # Initial parameters: regular polygon configuration
    outer_angles = np.zeros(outer_n)
    inner_angles = np.zeros(inner_n)
    outer_r = 1.0
    inner_r = 0.5
    ex = 1.0
    ey = 1.0
    outer_pert = 0.0
    inner_pert = 0.0
    
    x0 = np.concatenate([outer_angles, inner_angles,
                          [outer_r, inner_r, ex, ey, outer_pert, inner_pert]])
    
    best_params = x0.copy()
    best_val = _soft_obj(pts_to_flat(_two_ring_params_to_pts(outer_n, inner_n, x0)), beta=100.0)
    
    # Try multiple beta values
    for beta in [50, 100, 200, 400]:
        if time.time() - start_time > time_budget - 2:
            break
        
        def obj_beta(params):
            pts = _two_ring_params_to_pts(outer_n, inner_n, params)
            return _soft_obj(pts.flatten(), beta=beta)
        
        try:
            res = minimize(obj_beta, best_params, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11,
                                  'adaptive': True, 'maxfev': 5000})
            if res.x is not None:
                val = _soft_obj(_two_ring_params_to_pts(outer_n, inner_n, res.x).flatten(), beta=beta)
                if val < best_val:
                    best_val = val
                    best_params = res.x
        except Exception:
            pass
    
    pts = _two_ring_params_to_pts(outer_n, inner_n, best_params)
    return pts


def pts_to_flat(pts):
    return pts.flatten()


def _optimize_full_points(time_budget, init_pts):
    """
    Optimize all 28 coordinates directly with Nelder-Mead on the soft objective.
    """
    start_time = time.time()
    x0 = init_pts.flatten()
    
    best_x = x0.copy()
    best_score = _score(init_pts)
    
    # Stage 1: Soft objective with increasing beta
    for beta in [30, 60, 120, 250, 500]:
        if time.time() - start_time > time_budget - 3:
            break
        
        def obj_func(x):
            return _soft_obj(x, beta=beta)
        
        try:
            res = minimize(obj_func, best_x, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11,
                                  'adaptive': True, 'maxfev': 4000})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_x = res.x
        except Exception:
            pass
    
    # Stage 2: Exact score polishing
    if time.time() - start_time < time_budget - 2:
        try:
            res = minimize(_neg_score, best_x, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12,
                                  'adaptive': True, 'maxfev': 8000})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_x = res.x
        except Exception:
            pass
    
    return best_x.reshape(-1, 2)


def _initial_regular_polygon(n):
    """Points on a regular n-gon inscribed in unit circle, scaled to unit area."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _normalize_unit_area(pts)


def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 0:
        return pts / np.sqrt(hull_a)
    return pts


def _initial_two_ring(n):
    """Two-ring initialization: 10 outer + 4 inner."""
    outer_n = 10
    inner_n = n - outer_n
    outer_angles = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    inner = 0.5 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    pts = np.vstack([outer, inner])
    return _normalize_unit_area(pts)


def _initial_perturbed_polygon(n, seed=42):
    """Regular polygon with small random perturbations."""
    rng = np.random.RandomState(seed)
    angles = 2 * np.pi * np.arange(n) / n + rng.normal(0, 0.05, n)
    radii = 1.0 + rng.normal(0, 0.05, n)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return _normalize_unit_area(pts)


def _polar_to_cartesian(params):
    """
    Convert polar parameters to Cartesian points.
    params: [r_1...r_N, theta_1, d_1...d_{N-1}]
    theta_k = theta_1 + sum(d_1...d_{k-1})
    """
    n = N
    radii = params[:n]
    theta_start = params[n]
    d_thetas = params[n+1:n+n-1]
    
    thetas = np.zeros(n)
    thetas[0] = theta_start
    if n > 1:
        thetas[1:] = theta_start + np.cumsum(d_thetas)
        
    # Ensure thetas are within [0, 2*pi) for consistency, though not strictly necessary
    thetas = thetas % (2 * np.pi)
    
    x = radii * np.cos(thetas)
    y = radii * np.sin(thetas)
    return np.column_stack([x, y])

def _neg_score_polar(params):
    """
    Objective function for Differential Evolution.
    Minimizes negative score.
    """
    try:
        pts = _polar_to_cartesian(params)
        if not np.all(np.isfinite(pts)):
            return 1e10
        s = _score(pts)
        return -s
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    
    # Generate diverse initial configurations to establish a baseline
    inits = [
        _initial_regular_polygon(N),
        _initial_two_ring(N),
        _initial_perturbed_polygon(N, seed=1),
        _initial_perturbed_polygon(N, seed=2),
        _initial_perturbed_polygon(N, seed=3),
    ]
    
    # Evaluate initial scores
    init_scores = []
    for init in inits:
        s = _score(init)
        init_scores.append(s)
    
    # Sort by initial score (best first)
    sorted_indices = np.argsort(init_scores)[::-1]
    inits = [inits[i] for i in sorted_indices]
    
    best_pts = inits[0]
    best_score = init_scores[sorted_indices[0]]
    
    # Time allocation
    # We will allocate most time to Differential Evolution, which is global.
    # Then a small amount to Nelder-Mead polish.
    
    de_time_budget = 100.0  # 100 seconds for DE
    polish_time_budget = 20.0 # 20 seconds for polish
    
    # 1. Differential Evolution Search
    # Parameters: r_i (14), theta_1 (1), d_thetas (13) -> 28 params
    # Bounds:
    # r_i: [0.1, 2.0]
    # theta_1: [0, 2*pi]
    # d_thetas: [0.1, 2*pi] (ensure positive angular steps)
    
    bounds = []
    for _ in range(N):
        bounds.append((0.1, 2.0))
    bounds.append((0, 2 * np.pi))
    for _ in range(N - 1):
        bounds.append((0.1, 2 * np.pi))
        
    # We use a wrapper to check time
    de_start = time.time()
    try:
        # differential_evolution minimizes
        # maxiter is number of generations
        # popsize is multiplier of dimension
        # We want to use as much time as possible.
        # DE calls the objective function many times.
        # Let's try maxiter=50, popsize=15.
        # 28 dims * 15 pop = 420 points per gen.
        # 50 gens = 21000 evals.
        # Each eval is ~0.1-0.5ms? 21000 * 0.5ms = 10.5s.
        # We can afford more.
        
        # Let's use a callback to stop if time runs out.
        stop_flag = [False]
        
        def de_callback(xk, convergence):
            if time.time() - de_start > de_time_budget:
                stop_flag[0] = True
            return stop_flag[0]
            
        res = differential_evolution(
            _neg_score_polar, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-4, 
            seed=42,
            callback=de_callback,
            polish=False # We will do our own polish
        )
        
        if res.x is not None:
            cand_pts = _polar_to_cartesian(res.x)
            s = _score(cand_pts)
            if s > best_score:
                best_score = s
                best_pts = cand_pts
                
    except Exception:
        pass
    
    # 2. Nelder-Mead Polish on the best result found so far
    if time.time() - start_time < DEADLINE - 5:
        polish_start = time.time()
        polish_budget = min(polish_time_budget, DEADLINE - 5 - (time.time() - start_time))
        
        if polish_budget > 1:
            try:
                # Use the soft objective for smoothness during polish
                def final_obj(x):
                    return _soft_obj(x, beta=500.0)
                
                res = minimize(final_obj, best_pts.flatten(), method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                                      'adaptive': True, 'maxfev': 4000})
                if res.x is not None:
                    cand = res.x.reshape(-1, 2)
                    s = _score(cand)
                    if s > best_score:
                        best_score = s
                        best_pts = cand
            except Exception:
                pass
            
            # One more exact score polish
            if time.time() - polish_start < polish_budget - 2:
                try:
                    res = minimize(_neg_score, best_pts.flatten(), method='Nelder-Mead',
                                  options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12,
                                          'adaptive': True, 'maxfev': 2000})
                    if res.x is not None:
                        cand = res.x.reshape(-1, 2)
                        s = _score(cand)
                        if s > best_score:
                            best_score = s
                            best_pts = cand
                except Exception:
                    pass

    if best_pts is None:
        best_pts = _initial_regular_polygon(N)
    
    return best_pts
