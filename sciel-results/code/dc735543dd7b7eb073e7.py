"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

DEADLINE = 135.0  # Safety margin before 150s kill


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


def _param_to_points(params):
    """Convert (r, theta) polar params to Cartesian points.
    params: shape (N, 2) where each row is (radius, angle)
    Angles are sorted to enforce angular order.
    """
    radii = params[:, 0]
    angles = params[:, 1]
    # Sort by angle to enforce angular order
    order = np.argsort(angles)
    radii_sorted = radii[order]
    angles_sorted = angles[order]
    x = radii_sorted * np.cos(angles_sorted)
    y = radii_sorted * np.sin(angles_sorted)
    return np.column_stack([x, y])


def _param_score(params):
    """Score from polar parameters."""
    pts = _param_to_points(params)
    return _score(pts)


def _param_neg_score(params_flat):
    """Negative score for minimization from polar params."""
    params = params_flat.reshape(N, 2)
    s = _param_score(params)
    return -s


def _param_soft_obj(params_flat, beta=150.0):
    """Soft objective in polar parameter space."""
    params = params_flat.reshape(N, 2)
    pts = _param_to_points(params)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_triangle_areas(pts)
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
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val


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
    pts = []
    pts.append([0.0, 0.0])
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append([np.cos(angle), np.sin(angle)])
    for i in range(7):
        angle = 2 * np.pi * i / 7 + np.pi / 7
        r = 1.8
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    
    pts = np.array(pts)
    return _normalize_unit_area(pts)


def _initial_two_ring(n):
    """Two concentric rings with alternating angles."""
    outer_n = 8
    inner_n = n - outer_n
    angles_out = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    
    angles_in = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    r_in = 0.55
    inner = r_in * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
    
    pts = np.vstack([outer, inner])
    return _normalize_unit_area(pts)


def _initial_three_ring(n):
    """Three concentric rings."""
    ring_sizes = [6, 5, 3]
    ring_radii = [1.0, 0.6, 0.25]
    pts = []
    for size, r in zip(ring_sizes, ring_radii):
        angles = 2 * np.pi * np.arange(size) / size
        ring = r * np.column_stack([np.cos(angles), np.sin(angles)])
        pts.append(ring)
    pts = np.vstack(pts)
    return _normalize_unit_area(pts)


def _initial_random_well_spaced(n, seed=123):
    """Random points with minimum distance constraint."""
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
        return _initial_regular_polygon(n)
    
    pts = np.array(pts)
    return _normalize_unit_area(pts)


def _cartesian_to_polar(pts):
    """Convert Cartesian points to (radius, angle) parameters."""
    r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    # Ensure angles are in [0, 2*pi)
    theta = np.mod(theta, 2 * np.pi)
    return np.column_stack([r, theta])


def construct():
    start_time = time.time()
    
    # Initial configurations in Cartesian space
    inits_cart = [
        _initial_regular_polygon(N),
        _initial_mixed(N),
        _initial_hexagonal(N),
        _initial_two_ring(N),
        _initial_three_ring(N),
        _initial_random_well_spaced(N, seed=123),
        _initial_random_well_spaced(N, seed=456),
    ]
    
    # Add perturbed regular polygons
    rng = np.random.RandomState(42)
    for _ in range(3):
        base = _initial_regular_polygon(N)
        noise = rng.normal(0, 0.05, size=base.shape)
        perturbed = base + noise
        perturbed = _normalize_unit_area(perturbed)
        inits_cart.append(perturbed)
    
    best_pts = None
    best_score = 0.0
    
    def _update_best(cand_pts):
        nonlocal best_pts, best_score
        s = _score(cand_pts)
        if s > best_score:
            best_score = s
            best_pts = cand_pts
            return True
        return False
    
    def _try_optimize_cart(x0, method, obj_func, options, max_time=20):
        """Try Cartesian optimization."""
        t0 = time.time()
        try:
            res = minimize(obj_func, x0, method=method, options=options)
            if res.x is not None and time.time() - t0 < max_time:
                cand = res.x.reshape(-1, 2)
                _update_best(cand)
        except Exception:
            pass
    
    def _try_optimize_polar(x0, method, obj_func, options, max_time=20):
        """Try polar parameter optimization."""
        t0 = time.time()
        try:
            res = minimize(obj_func, x0, method=method, options=options)
            if res.x is not None and time.time() - t0 < max_time:
                params = res.x.reshape(N, 2)
                cand = _param_to_points(params)
                _update_best(cand)
        except Exception:
            pass
    
    # Phase 1: Quick Cartesian optimizations from multiple starts
    for init in inits_cart:
        if time.time() - start_time > DEADLINE - 40:
            break
        
        x0 = init.flatten()
        
        # L-BFGS-B with soft objective
        if time.time() - start_time < DEADLINE - 35:
            for beta in [80.0, 150.0]:
                if time.time() - start_time > DEADLINE - 35:
                    break
                _try_optimize_cart(
                    x0, 'L-BFGS-B',
                    lambda x, b=beta: _soft_obj(x, b),
                    {'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 1500},
                    max_time=15
                )
                if best_pts is not None:
                    x0 = best_pts.flatten()
        
        # Nelder-Mead with hard objective
        if time.time() - start_time < DEADLINE - 35:
            _try_optimize_cart(
                x0, 'Nelder-Mead',
                _neg_score,
                {'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True},
                max_time=15
            )
    
    # Phase 2: Differential Evolution in polar parameter space
    if time.time() - start_time < DEADLINE - 50:
        # Define bounds for polar parameters
        # Radii: [0, 2.0], Angles: [0, 2*pi)
        bounds = [(0.0, 2.0)] * N + [(0.0, 2 * np.pi)] * N
        bounds = list(zip(bounds[:N], bounds[N:]))
        
        # Run differential evolution
        try:
            de_result = differential_evolution(
                _param_neg_score,
                bounds=bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=True,
                init='sobol',
                update_interval=10,
                callback=None
            )
            if de_result.x is not None:
                params = de_result.x.reshape(N, 2)
                cand = _param_to_points(params)
                _update_best(cand)
        except Exception:
            pass
    
    # Phase 3: Refine best with polar parameter optimization
    if best_pts is not None and time.time() - start_time < DEADLINE - 30:
        params0 = _cartesian_to_polar(best_pts).flatten()
        
        # L-BFGS-B in polar space
        if time.time() - start_time < DEADLINE - 25:
            for beta in [100.0, 200.0]:
                if time.time() - start_time > DEADLINE - 25:
                    break
                _try_optimize_polar(
                    params0, 'L-BFGS-B',
                    lambda x, b=beta: _param_soft_obj(x, b),
                    {'maxiter': 400, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000},
                    max_time=15
                )
                if best_pts is not None:
                    params0 = _cartesian_to_polar(best_pts).flatten()
        
        # Nelder-Mead in polar space
        if time.time() - start_time < DEADLINE - 25:
            _try_optimize_polar(
                params0, 'Nelder-Mead',
                _param_neg_score,
                {'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True},
                max_time=20
            )
    
    # Phase 4: Final Cartesian polish
    if best_pts is not None and time.time() - start_time < DEADLINE - 15:
        x0 = best_pts.flatten()
        
        # L-BFGS-B with high beta
        _try_optimize_cart(
            x0, 'L-BFGS-B',
            lambda x: _soft_obj(x, beta=300.0),
            {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000},
            max_time=15
        )
        
        # Nelder-Mead with very tight tolerance
        if time.time() - start_time < DEADLINE - 10:
            _try_optimize_cart(
                x0, 'Nelder-Mead',
                _neg_score,
                {'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True},
                max_time=20
            )
    
    # Phase 5: Second DE run if time permits
    if time.time() - start_time < DEADLINE - 25:
        try:
            bounds = [(0.0, 2.0)] * N + [(0.0, 2 * np.pi)] * N
            bounds = list(zip(bounds[:N], bounds[N:]))
            
            de_result2 = differential_evolution(
                _param_neg_score,
                bounds=bounds,
                seed=789,
                maxiter=30,
                popsize=12,
                tol=1e-6,
                polish=True,
                init='sobol',
                update_interval=10,
                callback=None
            )
            if de_result2.x is not None:
                params = de_result2.x.reshape(N, 2)
                cand = _param_to_points(params)
                _update_best(cand)
        except Exception:
            pass
    
    # Final polish after second DE
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        x0 = best_pts.flatten()
        _try_optimize_cart(
            x0, 'Nelder-Mead',
            _neg_score,
            {'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True},
            max_time=15
        )
    
    if best_pts is None:
        best_pts = _initial_regular_polygon(N)
    
    return best_pts
# EVOLVE-BLOCK-END
