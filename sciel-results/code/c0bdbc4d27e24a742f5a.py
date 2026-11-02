"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

# ---------- Core geometry ----------

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points (returns min area, not doubled)."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area_fast(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)

def _objective_hard(points_flat):
    """Minimize hull_area / min_triangle_area (scale-invariant)."""
    points = points_flat.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    min_area = _min_triangle_area_vec(points)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area

def _objective_soft(points_flat, beta=150.0):
    """Smooth soft-min objective: hull_area / soft_min(triangle areas)."""
    points = points_flat.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
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
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val

def _normalize_unit_area(points):
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area > 1e-10:
            scale = np.sqrt(1.0 / hull_area)
            return points * scale
    except Exception:
        pass
    xs = points[:, 0]
    ys = points[:, 1]
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        scale = 1.0 / np.sqrt(w * h)
        return points * scale
    return points

# ---------- Centrally symmetric parameterization ----------
# For N=14, we use 7 independent vertex pairs (v_i, -v_i).
# Parameterize by 7 angles and 7 radii: v_i = r_i * (cos(theta_i), sin(theta_i))
# This ensures central symmetry by construction.

def _sym_points_to_params(points):
    """Extract 7 angles and 7 radii from a centrally symmetric point set."""
    # Assume first 7 points are one half, last 7 are negatives
    half = points[:7]
    radii = np.linalg.norm(half, axis=1)
    radii = np.maximum(radii, 1e-8)
    angles = np.arctan2(half[:, 1], half[:, 0])
    # Sort by angle to ensure convex ordering
    order = np.argsort(angles)
    return angles[order], radii[order]

def _params_to_points(angles, radii):
    """Generate 14 points from 7 angles and 7 radii (centrally symmetric)."""
    half = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return np.vstack([half, -half])

def _params_to_points_flat(params):
    """Decode 14-parameter vector [angles(7), radii(7)] to 14 points."""
    angles = params[:7]
    radii = params[7:]
    radii = np.maximum(radii, 1e-8)
    return _params_to_points(angles, radii)

def _objective_sym(params):
    """Hard objective for centrally symmetric parameters."""
    if np.any(~np.isfinite(params)):
        return 1e10
    points = _params_to_points_flat(params)
    if np.any(~np.isfinite(points)):
        return 1e10
    min_area = _min_triangle_area_vec(points)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area

def _objective_sym_soft(params, beta=150.0):
    """Soft objective for centrally symmetric parameters."""
    if np.any(~np.isfinite(params)):
        return 1e10
    points = _params_to_points_flat(params)
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
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val

def _generate_sym_inits():
    """Generate initial parameter vectors for centrally symmetric optimization."""
    inits = []
    
    # Regular heptagon-based (7 equally spaced angles)
    base_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    inits.append(np.concatenate([base_angles, np.ones(7)]))
    
    # Rotated versions
    for rot in [0.1, 0.3, 0.5, 0.8, 1.2, 1.7, 2.3]:
        a = base_angles + rot
        inits.append(np.concatenate([a, np.ones(7)]))
    
    # Non-uniform angle patterns with varying radii
    rng = np.random.default_rng(42)
    for seed in range(25):
        rng = np.random.default_rng(seed * 11 + 5)
        angles = np.sort(rng.uniform(0, 2*np.pi, 7))
        radii = 1.0 + rng.normal(0, 0.15, 7)
        radii = np.maximum(radii, 0.5)
        inits.append(np.concatenate([angles, radii]))
    
    # Pattern: alternating radii
    for r_pattern in [
        [1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0],
        [1.0, 0.7, 0.9, 0.7, 1.0, 0.7, 0.9],
        [1.2, 0.9, 1.1, 0.9, 1.2, 0.9, 1.1],
    ]:
        a = base_angles + 0.2
        inits.append(np.concatenate([a, np.array(r_pattern)]))
    
    # Jittered regular
    for js in [0.05, 0.1, 0.2]:
        a = base_angles + rng.normal(0, js, 7)
        r = 1.0 + rng.normal(0, js*0.5, 7)
        r = np.maximum(r, 0.5)
        inits.append(np.concatenate([a, r]))
    
    return inits

# ---------- Optimizers ----------

def _nm(x0, deadline, max_iter=2500, obj=_objective_sym):
    remaining = deadline - time.time()
    if remaining < 2.0:
        return None
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        return res
    except Exception:
        return None

def _lbfgs(x0, deadline, max_iter=5000, obj=_objective_sym_soft):
    remaining = deadline - time.time()
    if remaining < 5.0:
        return None
    try:
        res = minimize(obj, x0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        return res
    except Exception:
        return None

def _powell(x0, deadline, max_iter=1500, obj=_objective_sym):
    remaining = deadline - time.time()
    if remaining < 3.0:
        return None
    try:
        res = minimize(obj, x0, method='Powell',
                       options={'maxiter': max_iter, 'xtol': 1e-11, 'ftol': 1e-13})
        return res
    except Exception:
        return None

def _de_sym(bounds, deadline, max_iter=300, popsize=15):
    """Differential evolution on symmetric parameters."""
    remaining = deadline - time.time()
    if remaining < 10.0:
        return None
    try:
        res = differential_evolution(_objective_sym, bounds, maxiter=max_iter,
                                      popsize=popsize, tol=1e-10, atol=1e-12,
                                      seed=42, polish=True, init='latinhypercube')
        return res
    except Exception:
        return None

# ---------- Main ----------

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    best_score = 0.0
    best_pts = None
    best_params = None

    def _consider_from_params(params):
        nonlocal best_score, best_pts, best_params
        points = _params_to_points_flat(params)
        if points.shape != (N, 2) or np.any(~np.isfinite(points)):
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_pts = points.copy()
            best_params = params.copy()

    def _consider_points(pts):
        nonlocal best_score, best_pts
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Phase 1: Differential evolution in symmetric parameter space (global search)
    bounds = [(-np.pi, np.pi)] * 7 + [(0.3, 2.0)] * 7
    res = _de_sym(bounds, deadline, max_iter=200, popsize=20)
    if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
        _consider_from_params(res.x)

    # Phase 2: Nelder-Mead from diverse symmetric starts
    inits = _generate_sym_inits()
    for x0 in inits:
        if time.time() > deadline - 40:
            break
        res = _nm(x0, deadline, max_iter=2000, obj=_objective_sym)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider_from_params(res.x)

    # Phase 3: L-BFGS on soft objective from best symmetric params
    if best_params is not None and time.time() < deadline - 25:
        res = _lbfgs(best_params, deadline, max_iter=3000, obj=_objective_sym_soft)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider_from_params(res.x)

    # Phase 4: Powell polish on symmetric params
    if best_params is not None and time.time() < deadline - 15:
        res = _powell(best_params, deadline, max_iter=2000, obj=_objective_sym)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider_from_params(res.x)

    # Phase 5: Final NM polish
    if best_params is not None and time.time() < deadline - 5:
        res = _nm(best_params, deadline, max_iter=3000, obj=_objective_sym)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider_from_params(res.x)

    # Phase 6: One more DE run with tighter bounds if time permits
    if time.time() < deadline - 15:
        bounds_tight = [(-np.pi, np.pi)] * 7 + [(0.5, 1.5)] * 7
        res = _de_sym(bounds_tight, deadline, max_iter=100, popsize=15)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider_from_params(res.x)

    # Fallback: regular 14-gon
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    final_points = _normalize_unit_area(best_pts)

    # Safety check
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-10:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))

    return final_points
# EVOLVE-BLOCK-END
