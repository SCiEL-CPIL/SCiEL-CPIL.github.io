"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

# ---------- Precompute triangle index triples ----------
def _precompute_indices():
    idxs = list(combinations(range(N), 3))
    return np.array(idxs)

_IDX = _precompute_indices()
_I = _IDX[:, 0]
_J = _IDX[:, 1]
_K = _IDX[:, 2]

_SQRT3 = np.sqrt(3)
_H = _SQRT3 / 2.0

def _min_triangle_area_vec(points):
    """Minimum area of any triangle formed by three of the N points. Vectorized."""
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return float(np.min(areas))

# ---------- Vectorized clamp to triangle ----------
def _clamp_all_vec(points):
    """Clamp all points to be inside the equilateral triangle. Fully vectorized."""
    pts = np.array(points, dtype=float).copy()
    x = pts[:, 0]
    y = pts[:, 1]
    eps = 1e-9
    y = np.clip(y, eps, _H - eps)
    x_min = y / _SQRT3 + eps
    x_max = 1.0 - y / _SQRT3 - eps
    x = np.clip(x, x_min, x_max)
    pts[:, 0] = x
    pts[:, 1] = y
    return pts

# ---------- Penalty (smooth) for gradient-based methods ----------
def _violation_sum(points):
    """Sum of squared boundary violations (0 when all inside)."""
    x = points[:, 0]
    y = points[:, 1]
    d1 = np.maximum(0.0, -y)
    d2 = np.maximum(0.0, y - _SQRT3 * x)
    d3 = np.maximum(0.0, y - _SQRT3 * (1.0 - x))
    return float(np.sum(d1 * d1) + np.sum(d2 * d2) + np.sum(d3 * d3))

def _objective_smooth(params):
    """Smooth objective: -min_area + penalty for being outside the triangle."""
    pts = params.reshape(N, 2)
    min_a = _min_triangle_area_vec(pts)
    pen = _violation_sum(pts)
    return -min_a + 1e4 * pen

def _objective_nm(params):
    """Nelder-Mead objective: clamps then returns -min_area."""
    pts = _clamp_all_vec(params.reshape(N, 2))
    return -_min_triangle_area_vec(pts)

# ---------- Simulated Annealing (basin exploration) ----------
def _simulated_annealing(start_points, time_limit=10.0, seed=None):
    rng = np.random.default_rng(seed)
    points = _clamp_all_vec(np.array(start_points, dtype=float))
    current_score = _min_triangle_area_vec(points)
    best_points = points.copy()
    best_score = current_score

    T = 0.01
    T_min = 1e-7
    alpha = 0.999
    step_size = 0.05

    start_time = time.time()
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
        idx = rng.integers(N)
        noise = rng.normal(0, step_size, 2)
        new_point = points[idx] + noise
        new_point = _clamp_all_vec(new_point.reshape(1, 2))[0]
        old_point = points[idx].copy()
        points[idx] = new_point
        new_score = _min_triangle_area_vec(points)
        delta = new_score - current_score
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            points[idx] = old_point
        T *= alpha
        step_size *= 0.9999
    return best_points, best_score

# ---------- Gradient refinement (L-BFGS-B / BFGS) ----------
def _gradient_refine(points, time_limit=10.0):
    """Refine with gradient-based method using a smooth penalized objective."""
    x0 = _clamp_all_vec(np.array(points, dtype=float)).flatten()
    start_time = time.time()
    try:
        res = minimize(
            _objective_smooth,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9,
                     'maxfun': 20000, 'maxls': 50},
        )
        if time.time() - start_time > time_limit:
            return None
        cand = _clamp_all_vec(res.x.reshape(N, 2))
        return cand
    except Exception:
        return None

# ---------- Coordinate descent polish ----------
def _coord_descent(points, time_limit=5.0):
    """Small-step coordinate descent to escape shallow local minima."""
    pts = _clamp_all_vec(np.array(points, dtype=float)).copy()
    best = pts.copy()
    best_score = _min_triangle_area_vec(pts)
    start_time = time.time()
    step = 0.01
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            base = pts[i].copy()
            cur = best_score
            for dx in (step, -step, 0.0):
                for dy in (step, -step, 0.0):
                    if dx == 0.0 and dy == 0.0:
                        continue
                    cand = pts.copy()
                    cand[i] = _clamp_all_vec((base + np.array([dx, dy])).reshape(1, 2))[0]
                    s = _min_triangle_area_vec(cand)
                    if s > cur + 1e-12:
                        cur = s
                        pts = cand
                        improved = True
                        if cur > best_score:
                            best_score = cur
                            best = pts.copy()
        step *= 0.7
    return best, best_score

# ---------- Nelder-Mead refinement ----------
def _nelder_mead_optimize(points, time_limit=5.0):
    x0 = _clamp_all_vec(np.array(points, dtype=float)).flatten()
    start_time = time.time()
    best_points = x0.reshape(N, 2).copy()
    best_score = _min_triangle_area_vec(best_points)
    try:
        res = minimize(
            _objective_nm,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 400, 'maxfev': 1500, 'xatol': 1e-8, 'fatol': 1e-10},
        )
        if time.time() - start_time > time_limit:
            return best_points, best_score
        cand = _clamp_all_vec(res.x.reshape(N, 2))
        s = _min_triangle_area_vec(cand)
        if s > best_score:
            best_score = s
            best_points = cand
    except Exception:
        pass
    return best_points, best_score

# ---------- Initial configurations ----------
def _generate_initial_configs():
    h = _H
    configs = []

    # Symmetric: vertices, centroid, side midpoints, inner ring
    sym = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h / 3],
        [0.25, h / 6],
        [0.75, h / 6],
        [0.5, h / 2],
        [0.125, h / 12],
        [0.875, h / 12],
        [0.375, h / 4],
        [0.625, h / 4],
    ])
    configs.append(sym)

    # Second symmetric variant
    sym2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h / 2],
        [0.25, h / 4],
        [0.75, h / 4],
        [0.5, h / 6],
        [0.25, h / 12],
        [0.75, h / 12],
        [0.125, h / 24],
        [0.875, h / 24],
    ])
    configs.append(sym2)

    # Grid-like
    grid = np.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
        [0.25, h / 4],
        [0.75, h / 4],
        [0.5, h / 2],
        [0.125, h / 8],
        [0.375, h / 8],
        [0.625, h / 8],
        [0.875, h / 8],
        [0.5, h / 6],
    ])
    configs.append(grid)

    # Random configs
    rng = np.random.default_rng(42)
    for i in range(12):
        pts = rng.random((N, 2))
        pts[:, 1] *= h
        pts = _clamp_all_vec(pts)
        configs.append(pts)
    return configs

# ---------- Top-level ----------
def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # 12s safety margin before 150s kill

    best_points = None
    best_score = -1.0

    def _update(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    configs = _generate_initial_configs()
    n_configs = len(configs)

    # Phase 1: Nelder-Mead on each config (fast, diverse basins)
    for i, config in enumerate(configs):
        if time.time() > deadline:
            break
        time_left = deadline - time.time()
        per = min(4.0, time_left / (n_configs - i))
        if per < 1.0:
            break
        try:
            opt_pts, score = _nelder_mead_optimize(config, time_limit=per)
            _update(opt_pts, score)
        except Exception:
            continue

    # Phase 2: SA on the best found so far (escape local basins)
    if best_points is not None and time.time() < deadline - 25:
        remaining = deadline - time.time() - 25
        if remaining > 3.0:
            try:
                opt_pts, score = _simulated_annealing(best_points,
                                                       time_limit=remaining, seed=999)
                _update(opt_pts, score)
            except Exception:
                pass

    # Phase 3: Gradient refinement (L-BFGS-B) on best
    if best_points is not None and time.time() < deadline - 10:
        remaining = deadline - time.time() - 10
        if remaining > 2.0:
            try:
                cand = _gradient_refine(best_points, time_limit=remaining)
                if cand is not None:
                    _update(cand, _min_triangle_area_vec(cand))
            except Exception:
                pass

    # Phase 4: Coordinate-descent polish (escape shallow minima)
    if best_points is not None and time.time() < deadline - 4:
        remaining = deadline - time.time() - 4
        if remaining > 1.0:
            try:
                opt_pts, score = _coord_descent(best_points, time_limit=remaining)
                _update(opt_pts, score)
            except Exception:
                pass

    # Phase 5: Final tight gradient polish
    if best_points is not None and time.time() < deadline - 1:
        remaining = deadline - time.time() - 1
        if remaining > 1.0:
            try:
                cand = _gradient_refine(best_points, time_limit=remaining)
                if cand is not None:
                    _update(cand, _min_triangle_area_vec(cand))
            except Exception:
                pass

    if best_points is None:
        best_points = _clamp_all_vec(configs[0])

    return best_points

# EVOLVE-BLOCK-END
