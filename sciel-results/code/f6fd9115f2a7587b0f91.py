"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

DEADLINE = 140.0  # Safety margin before 150s kill


def _hull_area(pts):
    """Compute convex hull area."""
    try:
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
    """Return array of all triangle areas (N choose 3)."""
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


def _soft_obj(pts_flat, beta=150.0):
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


def _soft_obj_and_grad(pts_flat, beta=150.0):
    """Soft objective with numerical gradient for L-BFGS-B."""
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10, np.zeros_like(pts_flat)
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10, np.zeros_like(pts_flat)
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10, np.zeros_like(pts_flat)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10, np.zeros_like(pts_flat)
    
    # Compute numerical gradient
    eps = 1e-7
    grad = np.zeros_like(pts_flat)
    for i in range(len(pts_flat)):
        pts_plus = pts_flat.copy()
        pts_plus[i] += eps
        pts_minus = pts_flat.copy()
        pts_minus[i] -= eps
        
        # Compute objective at perturbed points
        def _eval_obj(pf):
            p = pf.reshape(N, 2)
            if np.any(~np.isfinite(p)):
                return 1e10
            a = _all_triangle_areas(p)
            if len(a) == 0 or np.min(a) < 1e-12:
                return 1e10
            zz = -beta * a
            mz = np.max(zz)
            if mz < -500:
                ls = mz
            else:
                ls = mz + np.log(np.sum(np.exp(zz - mz)))
            smv = -ls / beta
            if smv < 1e-12:
                return 1e10
            ha = _hull_area(p)
            if ha < 1e-10:
                return 1e10
            return ha / smv
        
        f_plus = _eval_obj(pts_plus)
        f_minus = _eval_obj(pts_minus)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    
    return hull_area / soft_min_val, grad


def _multi_beta_soft_obj(pts_flat, betas=(50.0, 150.0, 300.0)):
    """Multi-beta soft objective: weighted combination of soft-mins."""
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    total = 0.0
    for beta in betas:
        z = -beta * areas
        max_z = np.max(z)
        if max_z < -500:
            log_sum = max_z
        else:
            log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
        soft_min_val = -log_sum / beta
        if soft_min_val < 1e-12:
            return 1e10
        total += hull_area / soft_min_val
    return total / len(betas)


def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 0:
        return pts / np.sqrt(hull_a)
    return pts


def _initial_regular_polygon(n):
    """Points on a regular n-gon inscribed in unit circle, scaled to unit area."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _normalize_unit_area(pts)


def _initial_mixed(n):
    """Mix of boundary and interior points for better minimum triangle area."""
    outer_n = 10
    inner_n = n - outer_n
    angles = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(angles), np.sin(angles)])
    
    inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    inner = 0.4 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    pts = np.vstack([outer, inner])
    return _normalize_unit_area(pts)


def _initial_hexagonal(n):
    """Hexagonal lattice inspired initialization."""
    # Create a hexagonal arrangement
    # 14 points: center + ring of 6 + ring of 7 (approx)
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # First ring (6 points)
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append([np.cos(angle), np.sin(angle)])
    # Second ring (7 points to make 14 total)
    for i in range(7):
        angle = 2 * np.pi * i / 7 + np.pi / 7
        r = 1.8
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    
    pts = np.array(pts)
    return _normalize_unit_area(pts)


def _initial_cosine_mod(n):
    """Regular polygon with cosine-modulated radii."""
    angles = 2 * np.pi * np.arange(n) / n
    # Modulate radii with a cosine wave
    mod = 1.0 + 0.1 * np.cos(3 * angles)
    pts = np.column_stack([mod * np.cos(angles), mod * np.sin(angles)])
    return _normalize_unit_area(pts)


def _initial_two_ring(n):
    """Two concentric rings with alternating angles for better distribution."""
    outer_n = 8
    inner_n = n - outer_n
    angles_out = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    
    angles_in = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    r_in = 0.55
    inner = r_in * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
    
    pts = np.vstack([outer, inner])
    return _normalize_unit_area(pts)


def _initial_random_well_spaced(n, seed=123):
    """Random points with minimum distance constraint (rejection sampling)."""
    rng = np.random.RandomState(seed)
    pts = []
    min_dist = 0.25
    max_attempts = 5000
    attempts = 0
    while len(pts) < n and attempts < max_attempts:
        p = rng.uniform(-1, 1, size=2)
        if np.linalg.norm(p) > 1.0:
            continue
        if all(np.linalg.norm(np.array(p) - np.array(q)) >= min_dist for q in pts):
            pts.append(p)
        attempts += 1
    
    if len(pts) < n:
        # Fill with regular polygon
        return _initial_regular_polygon(n)
    
    pts = np.array(pts)
    return _normalize_unit_area(pts)


def construct():
    start_time = time.time()
    
    # Try multiple initial configurations
    inits = [
        _initial_regular_polygon(N),
        _initial_mixed(N),
        _initial_hexagonal(N),
        _initial_cosine_mod(N),
        _initial_two_ring(N),
        _initial_random_well_spaced(N, seed=123),
    ]
    
    # Also try a few random perturbations of the regular polygon
    rng = np.random.RandomState(42)
    for _ in range(4):
        base = _initial_regular_polygon(N)
        noise = rng.normal(0, 0.04, size=base.shape)
        perturbed = base + noise
        perturbed = _normalize_unit_area(perturbed)
        inits.append(perturbed)
    
    best_pts = None
    best_score = 0.0
    
    def _try_optimize(x0, method, obj_func, options, max_time=30):
        """Try an optimization and return (points, score) if improved."""
        nonlocal best_pts, best_score
        t0 = time.time()
        try:
            res = minimize(obj_func, x0, method=method, options=options)
            if res.x is not None and time.time() - t0 < max_time:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    for init in inits:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        x0 = init.flatten()
        
        # Pass 1: L-BFGS-B with beta-annealing (gradient-based, faster convergence)
        if time.time() - start_time < DEADLINE - 10:
            for beta in [50.0, 100.0, 200.0]:
                if time.time() - start_time > DEADLINE - 10:
                    break
                _try_optimize(
                    x0, 'L-BFGS-B',
                    lambda x, b=beta: _soft_obj(x, b),
                    {'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000},
                    max_time=25
                )
                if best_pts is not None:
                    x0 = best_pts.flatten()
        
        # Pass 2: Nelder-Mead with multi-beta soft objective
        if time.time() - start_time < DEADLINE - 10:
            _try_optimize(
                x0, 'Nelder-Mead',
                _multi_beta_soft_obj,
                {'maxiter': 2500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True},
                max_time=25
            )
        
        # Pass 3: Nelder-Mead with single beta
        if time.time() - start_time < DEADLINE - 10:
            _try_optimize(
                x0, 'Nelder-Mead',
                lambda x: _soft_obj(x, beta=200.0),
                {'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True},
                max_time=20
            )
        
        # Pass 4: Hard objective for precise local optimization
        if time.time() - start_time < DEADLINE - 10:
            _try_optimize(
                x0, 'Nelder-Mead',
                _neg_score,
                {'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True},
                max_time=20
            )
    
    # Final polish with tighter tolerance if time permits
    if best_pts is not None and time.time() - start_time < DEADLINE - 20:
        x0 = best_pts.flatten()
        
        # L-BFGS-B polish with high beta
        _try_optimize(
            x0, 'L-BFGS-B',
            lambda x: _soft_obj(x, beta=300.0),
            {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000},
            max_time=25
        )
    
    # Nelder-Mead polish with very tight tolerance
    if best_pts is not None and time.time() - start_time < DEADLINE - 15:
        x0 = best_pts.flatten()
        _try_optimize(
            x0, 'Nelder-Mead',
            _neg_score,
            {'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True},
            max_time=30
        )
    
    # Additional polish with soft objective for robustness
    if best_pts is not None and time.time() - start_time < DEADLINE - 10:
        x0 = best_pts.flatten()
        _try_optimize(
            x0, 'Nelder-Mead',
            lambda x: _soft_obj(x, beta=400.0),
            {'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True},
            max_time=25
        )
    
    if best_pts is None:
        # Fallback: regular polygon
        best_pts = _initial_regular_polygon(N)
    
    return best_pts
# EVOLVE-BLOCK-END
