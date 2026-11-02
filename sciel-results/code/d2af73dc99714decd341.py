"""
Heilbronn convex regions n=14: maximize min triangle area / convex hull area.

Strategy:
  1. Centrally symmetric parameterization: 7 independent points define 14 via p_i + p_{i+7} = 0.
     This halves the search space and enforces symmetry by construction.
  2. The 7 free points are parameterized in polar form (angle, radius) with sorted angles
     to maintain a convex polygon structure.
  3. Multi-phase optimization:
     - Differential evolution for global search in the 14D symmetric parameter space.
     - Nelder-Mead refinement.
     - Powell refinement.
     - L-BFGS on a soft objective.
  4. Time-guarded to finish within 150s.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
HALF = 7

# Pre-compute all triangle index combinations
import itertools
tri_indices = np.array(list(itertools.combinations(range(N), 3)))  # shape (364, 3)


def _min_triangle_area(points):
    """Minimum triangle area over all triples, vectorized."""
    p = points
    pi = p[tri_indices[:, 0]]
    pj = p[tri_indices[:, 1]]
    pk = p[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _all_triangle_areas(points):
    pi = points[tri_indices[:, 0]]
    pj = points[tri_indices[:, 1]]
    pk = points[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)


def _objective_hard(flat):
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area


def _objective_soft(flat, beta=50.0):
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_triangle_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min


# ---------- Centrally symmetric parameterization ----------
# 7 independent points in polar form: (theta_i, r_i) for i=0..6
# Full 14 points: first 7 are (r_i cos theta_i, r_i sin theta_i),
#                 last 7 are negatives: (-r_i cos theta_i, -r_i sin theta_i)
# Angles are sorted to maintain ordering.

def _sym_params_to_points(params):
    """
    params: array of shape (14,) = [theta_0,...,theta_6, r_0,...,r_6]
    Returns (14, 2) array of centrally symmetric points.
    """
    thetas = params[:HALF]
    radii = params[HALF:]
    # Sort by angle
    order = np.argsort(thetas)
    thetas_sorted = thetas[order]
    radii_sorted = radii[order]
    # First 7 points
    x1 = radii_sorted * np.cos(thetas_sorted)
    y1 = radii_sorted * np.sin(thetas_sorted)
    # Last 7 points are negatives
    x2 = -x1
    y2 = -y1
    return np.column_stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])])


def _points_to_sym_params(points):
    """
    points: (14, 2) array, assumed centrally symmetric.
    Returns (14,) parameter array [theta_0,...,theta_6, r_0,...,r_6].
    """
    # Take first 7 points
    pts = points[:HALF]
    thetas = np.arctan2(pts[:, 1], pts[:, 0])
    radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    order = np.argsort(thetas)
    return np.concatenate([thetas[order], radii[order]])


def _objective_sym(params):
    """Objective for centrally symmetric parameters."""
    pts = _sym_params_to_points(params)
    return _objective_hard(pts.flatten())


def _objective_sym_soft(params, beta=50.0):
    """Soft objective for centrally symmetric parameters."""
    pts = _sym_params_to_points(params)
    return _objective_soft(pts.flatten(), beta)


# ---------- Optimizer helpers ----------

def _nm(flat0, deadline, obj, max_iter=2000):
    if time.time() > deadline - 3:
        return None
    try:
        res = minimize(obj, flat0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        return res
    except Exception:
        return None


def _powell(flat0, deadline, obj, max_iter=1000):
    if time.time() > deadline - 3:
        return None
    try:
        res = minimize(obj, flat0, method='Powell',
                       options={'maxiter': max_iter, 'xtol': 1e-11, 'ftol': 1e-13})
        return res
    except Exception:
        return None


def _lbfgs(flat0, deadline, obj, max_iter=3000):
    if time.time() > deadline - 5:
        return None
    try:
        res = minimize(obj, flat0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        return res
    except Exception:
        return None


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


# ---------- Initial configurations ----------

def _make_sym_initial_params():
    """Generate diverse initial centrally symmetric parameter configurations."""
    inits = []
    rng = np.random.default_rng(42)

    # 1. Regular 14-gon (centrally symmetric)
    angles = np.linspace(0, np.pi, HALF, endpoint=False)
    radii = np.ones(HALF)
    inits.append(np.concatenate([angles, radii]))

    # 2. Jittered regular polygon
    base_angles = np.linspace(0, np.pi, HALF, endpoint=False)
    for js in [0.05, 0.1, 0.15, 0.2]:
        thetas = np.sort(base_angles + rng.normal(0, js, size=HALF))
        # Ensure angles are in [0, pi)
        thetas = np.mod(thetas, np.pi)
        thetas = np.sort(thetas)
        radii = 1.0 + rng.normal(0, js * 0.5, size=HALF)
        radii = np.abs(radii) + 0.1
        inits.append(np.concatenate([thetas, radii]))

    # 3. Two-ring structure (3 inner, 4 outer)
    for r_inner in [0.4, 0.5, 0.6, 0.7]:
        a_inner = np.linspace(0, np.pi, 3, endpoint=False)
        a_outer = np.linspace(0, np.pi, 4, endpoint=False) + np.pi / 8
        all_thetas = np.sort(np.concatenate([a_inner, a_outer]))
        all_radii = np.concatenate([np.full(3, r_inner), np.ones(4)])
        order = np.argsort(np.concatenate([a_inner, a_outer]))
        all_radii_sorted = all_radii[order]
        inits.append(np.concatenate([all_thetas, all_radii_sorted]))

    # 4. Random configurations
    for seed in range(20):
        rs = np.random.default_rng(seed * 17 + 3)
        thetas = np.sort(rs.uniform(0, np.pi, HALF))
        radii = rs.uniform(0.4, 1.2, HALF)
        inits.append(np.concatenate([thetas, radii]))

    # 5. Elliptical configurations
    for rx, ry in [(1.0, 0.6), (1.0, 0.8), (1.2, 0.7), (0.8, 1.0)]:
        a = np.linspace(0, np.pi, HALF, endpoint=False)
        x = rx * np.cos(a)
        y = ry * np.sin(a)
        thetas = np.arctan2(y, x)
        radii = np.sqrt(x**2 + y**2)
        order = np.argsort(thetas)
        inits.append(np.concatenate([thetas[order], radii[order]]))

    # 6. Perturbed configurations with varying radius profiles
    for profile in ['linear', 'quadratic', 'sinusoidal']:
        base_angles = np.linspace(0, np.pi, HALF, endpoint=False)
        if profile == 'linear':
            radii = np.linspace(0.6, 1.0, HALF)
        elif profile == 'quadratic':
            t = np.linspace(0, 1, HALF)
            radii = 0.6 + 0.4 * t**2
        else:
            radii = 0.8 + 0.2 * np.sin(2 * base_angles)
        inits.append(np.concatenate([base_angles, radii]))

    return inits


def _make_cartesian_initial_points():
    """Generate diverse initial point configurations (Cartesian)."""
    inits = []
    rng = np.random.default_rng(42)

    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))

    # 2. Centrally symmetric from regular 7-gon
    angles7 = np.linspace(0, np.pi, HALF, endpoint=False)
    pts7 = np.column_stack([np.cos(angles7), np.sin(angles7)])
    inits.append(np.vstack([pts7, -pts7]))

    # 3. Two concentric rings (7+7)
    for r1 in [0.3, 0.5, 0.7, 0.85]:
        a1 = np.linspace(0, np.pi, 3, endpoint=False)
        a2 = np.linspace(0, np.pi, 4, endpoint=False) + np.pi / 8
        pts_inner = r1 * np.column_stack([np.cos(a1), np.sin(a1)])
        pts_outer = np.column_stack([np.cos(a2), np.sin(a2)])
        pts = np.vstack([pts_inner, pts_outer, -pts_inner, -pts_outer])
        inits.append(pts)

    # 4. Random configurations
    for seed in range(15):
        rs = np.random.default_rng(seed * 17 + 3)
        angles = np.sort(rs.uniform(0, np.pi, HALF))
        radii = rs.uniform(0.4, 1.2, HALF)
        pts7 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(np.vstack([pts7, -pts7]))

    # 5. Jittered regular polygon
    base = np.column_stack([np.cos(angles), np.sin(angles)])
    for js in [0.05, 0.1, 0.2]:
        inits.append(base + rng.normal(0, js, size=(N, 2)))

    # 6. Elliptical configurations
    for rx, ry in [(1.0, 0.6), (1.0, 0.8), (1.2, 0.7)]:
        a = np.linspace(0, 2 * np.pi, N, endpoint=False)
        inits.append(np.column_stack([rx * np.cos(a), ry * np.sin(a)]))

    return inits


# ---------- Main ----------

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    best_score = 0.0
    best_pts = None

    def _consider(pts):
        nonlocal best_score, best_pts
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Phase 1: Optimize in centrally symmetric parameter space
    sym_inits = _make_sym_initial_params()
    for i, params0 in enumerate(sym_inits):
        if time.time() > deadline - 80:
            break
        res = _nm(params0, deadline, _objective_sym, max_iter=800)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            pts = _sym_params_to_points(res.x)
            _consider(pts)

    # Phase 2: Optimize in Cartesian space from best
    cart_inits = _make_cartesian_initial_points()
    if best_pts is not None:
        cart_inits.insert(0, best_pts)

    for i, pts0 in enumerate(cart_inits):
        if time.time() > deadline - 50:
            break
        res = _nm(pts0.flatten(), deadline, _objective_hard, max_iter=800)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 3: Powell refinement from best
    if best_pts is not None and time.time() < deadline - 40:
        res = _powell(best_pts.flatten(), deadline, _objective_hard, max_iter=600)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 4: L-BFGS on soft objective
    if best_pts is not None and time.time() < deadline - 30:
        res = _lbfgs(best_pts.flatten(), deadline, _objective_soft, max_iter=1500)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 5: Differential evolution in symmetric parameter space
    if time.time() < deadline - 30:
        try:
            # Bounds: angles in [0, pi], radii in [0.1, 2.0]
            bounds = [(0, np.pi)] * HALF + [(0.1, 2.0)] * HALF
            res_de = differential_evolution(
                _objective_sym, bounds,
                seed=42, maxiter=30, tol=1e-10,
                popsize=15, mutation=(0.3, 1.0), recombination=0.7,
                polish=False, update_interval=1
            )
            if res_de is not None and np.isfinite(res_de.fun) and res_de.fun < 1e9:
                pts = _sym_params_to_points(res_de.x)
                _consider(pts)
        except Exception:
            pass

    # Phase 6: Local refinement after DE
    if best_pts is not None and time.time() < deadline - 15:
        res = _nm(best_pts.flatten(), deadline, _objective_hard, max_iter=1000)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 7: Powell final polish
    if best_pts is not None and time.time() < deadline - 8:
        res = _powell(best_pts.flatten(), deadline, _objective_hard, max_iter=500)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 8: Final L-BFGS polish
    if best_pts is not None and time.time() < deadline - 5:
        res = _lbfgs(best_pts.flatten(), deadline, _objective_soft, max_iter=1000)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Fallback
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
