"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, minimize_scalar

N = 13

# EVOLVE-BLOCK-START

def _all_triangles(p):
    """Return a (286, 2) array: col0 = area, col1 = -1 if degenerate/collinear, else 1.
    We want all areas > 0 and maximize the minimum."""
    n = len(p)
    idx = list(itertools.combinations(range(n), 3))
    m = len(idx)
    areas = np.zeros(m)
    for t, (i, j, k) in enumerate(idx):
        a, b, c = p[i], p[j], p[k]
        areas[t] = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
    return areas

def _hull_area(p):
    """Compute convex hull area."""
    try:
        h = ConvexHull(p)
        return h.volume
    except Exception:
        return 0.0

def _score(p):
    """Score = min triangle area / hull area."""
    ha = _hull_area(p)
    if ha < 1e-14:
        return 0.0
    areas = _all_triangles(p)
    m = np.min(areas)
    if m < 1e-14:
        return 0.0
    return m / ha

def _normalize(p):
    """Normalize so hull area = 1."""
    ha = _hull_area(p)
    if ha < 1e-14:
        return p
    return p / np.sqrt(ha)

def _objective(x, deadline):
    """Objective for L-BFGS-B: maximize score (minimize -score)."""
    if time.time() > deadline:
        return 1e10
    p = x.reshape(N, 2)
    pn = _normalize(p)
    s = _score(pn)
    if s < 1e-14:
        return 1e10
    return -s

def _soft_min_obj(x, tau, deadline):
    """Soft-min surrogate: -logsumexp(areas/tau) / hull_area.
    This is smooth and allows L-BFGS-B to work."""
    if time.time() > deadline:
        return 1e10
    p = x.reshape(N, 2)
    ha = _hull_area(p)
    if ha < 1e-14:
        return 1e10
    areas = _all_triangles(p)
    # Soft min: -tau * log(sum(exp(-areas/tau)))
    logsum = np.log(np.sum(np.exp(-areas / tau)))
    soft_min = -tau * logsum
    if soft_min < 1e-14:
        return 1e10
    return -(soft_min / ha)

def _regular_polygon(n, r=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])

def _grid_13():
    """3x3 grid + center + 4 extra points."""
    pts = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            pts.append([i, j])
    # That's 9 points. Add 4 more: corners of a larger square
    pts.append([1.5, 1.5])
    pts.append([1.5, -1.5])
    pts.append([-1.5, 1.5])
    pts.append([-1.5, -1.5])
    return np.array(pts[:13], dtype=float)

def _circle_plus_center():
    """12 points on circle + 1 at center."""
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    center = np.array([0.0, 0.0])
    return np.vstack([pts, center.reshape(1, 2)])

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    rng = np.random.default_rng(12345)

    best_points = None
    best_score = 0.0

    # Initial configurations
    configs = [
        _regular_polygon(13),
        _grid_13(),
        _circle_plus_center(),
        rng.uniform(-1, 1, (N, 2)),
        rng.uniform(-1, 1, (N, 2)),
    ]

    for cfg in configs:
        if time.time() > deadline:
            break
        cfg = _normalize(cfg)
        s = _score(cfg)
        if s > best_score:
            best_score = s
            best_points = cfg.copy()

    if best_points is None:
        best_points = _regular_polygon(13)

    # Soft-min annealing with L-BFGS-B
    # Start with a large tau (smooth) and decrease
    taus = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    current = best_points.copy()

    for tau in taus:
        if time.time() > deadline - 10:
            break
        x0 = current.flatten()
        try:
            result = minimize(
                _soft_min_obj,
                x0,
                args=(tau, deadline),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
            )
            if result.fun < -best_score + 1e-8:  # Improved
                new_pts = _normalize(result.x.reshape(N, 2))
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_pts
                    current = new_pts
        except Exception:
            pass

    # Final hard-objective refinement with Nelder-Mead
    if best_points is not None and time.time() < deadline - 5:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _objective,
                x0,
                args=(deadline,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'disp': False}
            )
            new_pts = _normalize(result.x.reshape(N, 2))
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_points = new_pts
        except Exception:
            pass

    # Return the best points found
    if best_points is None:
        return _regular_polygon(13)

    return best_points

# EVOLVE-BLOCK-END
