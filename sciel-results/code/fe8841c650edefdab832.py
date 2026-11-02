"""Heilbronn 11 triangles: LP-based optimization using scipy.linprog with bisection on minimum area."""
import numpy as np
import time
import itertools
from scipy.optimize import linprog

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
_NTRIP = len(_TRIPLETS)

# Barycentric to Cartesian conversion
# Vertices: A=(0,0), B=(1,0), C=(0.5, S3/2)
# P = u*(1,0) + v*(0.5, S3/2) = (u + 0.5v, v*S3/2)

def _bary_to_cart(bary):
    """Convert barycentric coordinates (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    x = u + 0.5 * v
    y = v * S3H
    return np.column_stack((x, y))

def _min_triangle_area_cartesian(pts):
    """
    Compute the minimum area of any triangle formed by three points.
    Input: pts is an (N, 2) array of Cartesian coordinates.
    Returns the raw area (not normalized).
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

def _min_triangle_area_bary(bary):
    """
    Compute the minimum area of any triangle formed by three points given in barycentric coordinates.
    Area = TRI_AREA * |det([[1, u1, v1], [1, u2, v2], [1, u3, v3]])|
    """
    u = bary[:, 0]
    v = bary[:, 1]
    
    u1 = u[_T_I]
    v1 = v[_T_I]
    u2 = u[_T_J]
    v2 = v[_T_J]
    u3 = u[_T_K]
    v3 = v[_T_K]
    
    dets = (u2*v3 - u3*v2) - u1*(v3 - v2) + v1*(u3 - u2)
    areas = TRI_AREA * np.abs(dets)
    return np.min(areas)

def _project_to_bary(bary):
    """Ensure barycentric coordinates are within the triangle (u>=0, v>=0, u+v<=1)."""
    u = bary[:, 0]
    v = bary[:, 1]
    u_clipped = np.maximum(0.0, u)
    v_clipped = np.maximum(0.0, v)
    sum_uv = u_clipped + v_clipped
    mask = sum_uv > 1
    if np.any(mask):
        scale = 1.0 / sum_uv[mask]
        u_clipped[mask] *= scale
        v_clipped[mask] *= scale
    return np.column_stack((u_clipped, v_clipped))

def _build_lp_for_area(target_area):
    """
    Build the LP to maximize the minimum triangle area, given a target minimum area.
    
    Variables: x = [u_0, v_0, u_1, v_1, ..., u_10, v_10] (22 variables)
    Plus one variable t for the minimum area.
    
    Actually, let's use a different approach: we can use scipy.optimize.linprog
    to maximize t subject to:
    - For each triplet (i,j,k): area(i,j,k) >= t
    - u_i >= 0, v_i >= 0, u_i + v_i <= 1 for all i
    
    The area of triangle (i,j,k) in barycentric coordinates is:
    TRI_AREA * |det| where det = u_j*v_k - u_k*v_j - u_i*(v_k - v_j) + v_i*(u_k - u_j)
    
    Since we want area >= t, and area = TRI_AREA * |det|, we have:
    TRI_AREA * |det| >= t
    => |det| >= t / TRI_AREA
    
    This is two linear constraints:
    det >= t/TRI_AREA  OR  det <= -t/TRI_AREA
    
    But this is an OR constraint, which is not linear. 
    
    Alternative approach: We can use a different formulation.
    
    Let's think differently. The area of triangle (i,j,k) can be written as:
    area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    
    In Cartesian coordinates:
    p_i = (u_i + 0.5*v_i, v_i * S3H)
    
    Let's define:
    x_i = u_i, y_i = v_i
    
    p_i = (x_i + 0.5*y_i, y_i * S3H)
    
    p_j - p_i = (x_j - x_i + 0.5*(y_j - y_i), (y_j - y_i)*S3H)
    p_k - p_i = (x_k - x_i + 0.5*(y_k - y_i), (y_k - y_i)*S3H)
    
    cross = (x_j - x_i + 0.5*(y_j - y_i)) * (y_k - y_i)*S3H - (y_j - y_i)*S3H * (x_k - x_i + 0.5*(y_k - y_i))
    
    Let a = x_j - x_i, b = y_j - y_i, c = x_k - x_i, d = y_k - y_i
    
    cross = (a + 0.5*b) * d * S3H - b * (c + 0.5*d) * S3H
           = S3H * [a*d + 0.5*b*d - b*c - 0.5*b*d]
           = S3H * [a*d - b*c]
           = S3H * [(x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i)]
    
    So area = 0.5 * |S3H * [(x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i)]|
             = S3H/2 * |det|
             = (S3/2)/2 * |det|
             = S3/4 * |det|
             = TRI_AREA * |det|
    
    This confirms the barycentric formula.
    
    So the constraint is:
    TRI_AREA * |det(i,j,k)| >= t
    
    where det(i,j,k) = (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i)
    
    Expanding:
    det = x_j*y_k - x_j*y_i - x_i*y_k + x_i*y_i - y_j*x_k + y_j*x_i + y_i*x_k - y_i*x_i
        = x_j*y_k - x_i*y_k - x_j*y_i + y_i*x_k - y_j*x_k + y_j*x_i - x_i*y_i + x_i*y_i
        = x_j*y_k - x_i*y_k - x_j*y_i + y_i*x_k - y_j*x_k + y_j*x_i
    
    Wait, let me be more careful:
    det = (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i)
        = x_j*y_k - x_j*y_i - x_i*y_k + x_i*y_i - y_j*x_k + y_j*x_i + y_i*x_k - y_i*x_i
        = x_j*y_k - x_j*y_i - x_i*y_k - y_j*x_k + y_j*x_i + y_i*x_k
        (the x_i*y_i and -y_i*x_i cancel)
    
    So det = x_j*y_k - x_j*y_i - x_i*y_k - y_j*x_k + y_j*x_i + y_i*x_k
    
    This is bilinear in the variables, so the constraint TRI_AREA * |det| >= t is not linear.
    
    Alternative approach: Use a different optimization method.
    
    Let's use a sequential quadratic programming or a non-linear optimization approach.
    
    Actually, let's use scipy.optimize.minimize with a different method.
    
    New approach: Use a penalty method or a barrier method.
    
    Or, use a different insight: the problem is to maximize the minimum area.
    We can use a bisection approach: for a given target area t, check if it's feasible.
    But the feasibility check is still non-linear due to the absolute value.
    
    Let's try a different approach: use scipy.optimize.differential_evolution
    or scipy.optimize.dual_annealing.
    
    Actually, let's use a combination of differential evolution and local optimization.
    """
    pass

def _minimize_negative_min_area(bary):
    """Objective function: negative of the minimum triangle area."""
    return -_min_triangle_area_bary(bary)

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_bary = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(bary):
        nonlocal best_bary, best_area
        bary = _project_to_bary(bary)
        area = _min_triangle_area_bary(bary)
        if area > best_area:
            best_area = area
            best_bary = bary.copy()
            return True
        return False

    def _generate_initial_configs():
        """Generate diverse starting points in barycentric coordinates."""
        configs = []
        
        # Symmetric base configuration
        base_sym = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
            [1/3, 1/3],
            [1/6, 1/6],
            [5/6, 1/6],
            [1/6, 5/6],
            [0.25, 0.25],
        ])
        configs.append(base_sym)

        # Random uniform points
        for _ in range(15):
            bary = rng.random((N, 2))
            mask = bary[:, 0] + bary[:, 1] > 1
            bary[mask, 0] = 1 - bary[mask, 0]
            bary[mask, 1] = 1 - bary[mask, 1]
            configs.append(bary)
            
        # Perturbed symmetric base
        for _ in range(10):
            pts = base_sym + rng.normal(0, 0.15, size=(N, 2))
            configs.append(_project_to_bary(pts))

        return configs

    # Phase 1: Differential Evolution
    from scipy.optimize import differential_evolution
    
    def _de_objective(x):
        bary = x.reshape(N, 2)
        bary = _project_to_bary(bary)
        return -_min_triangle_area_bary(bary)
    
    # Bounds: u in [0,1], v in [0,1] (we'll project to the triangle)
    bounds = [(0.0, 1.0)] * (2 * N)
    
    configs = _generate_initial_configs()
    
    # Run DE from multiple starting points
    de_runs = 3
    for run_idx in range(de_runs):
        if not _elapsed_ok(30):
            break
        
        # Use a different subset of configs for each run
        start_idx = run_idx * 5
        config = configs[start_idx % len(configs)]
        
        try:
            result = differential_evolution(
                _de_objective,
                bounds=bounds,
                x0=config.flatten(),
                maxiter=200,
                popsize=15,
                tol=1e-8,
                seed=42 + run_idx,
                polish=True,
                update_interval=1,
                callback=lambda x, convergence: None
            )
            bary = result.x.reshape(N, 2)
            _update_best(bary)
        except Exception:
            pass
    
    # Phase 2: Local optimization using Nelder-Mead
    if best_bary is not None and _elapsed_ok(20):
        from scipy.optimize import minimize
        
        def nm_obj(x):
            bary = x.reshape(N, 2)
            bary = _project_to_bary(bary)
            return -_min_triangle_area_bary(bary)
        
        # Multiple restarts of Nelder-Mead
        nm_restarts = 5
        for r in range(nm_restarts):
            if not _elapsed_ok(5):
                break
            
            # Perturb the best solution
            if r == 0:
                x0 = best_bary.flatten()
            else:
                x0 = best_bary.flatten() + rng.normal(0, 0.05, size=2*N)
            
            try:
                res = minimize(
                    nm_obj,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                bary = res.x.reshape(N, 2)
                _update_best(bary)
            except Exception:
                pass
    
    # Phase 3: Fine polish with smaller step sizes
    if best_bary is not None and _elapsed_ok(10):
        from scipy.optimize import minimize
        
        def nm_obj_fine(x):
            bary = x.reshape(N, 2)
            bary = _project_to_bary(bary)
            return -_min_triangle_area_bary(bary)
        
        try:
            res = minimize(
                nm_obj_fine,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            bary = res.x.reshape(N, 2)
            _update_best(bary)
        except Exception:
            pass

    if best_bary is None:
        bary = rng.random((N, 2))
        mask = bary[:, 0] + bary[:, 1] > 1
        bary[mask, 0] = 1 - bary[mask, 0]
        bary[mask, 1] = 1 - bary[mask, 1]
        best_bary = _project_to_bary(bary)

    return _bary_to_cart(best_bary)

def _project_to_cartesian(pts):
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

def _cart_to_bary(pts):
    """
    Convert Cartesian points to Barycentric coordinates.
    """
    u = np.zeros(N)
    v = np.zeros(N)
    for i in range(N):
        x, y = pts[i]
        v[i] = 2 * y / S3
        u[i] = x - 0.5 * v[i]
    return np.column_stack((u, v))
