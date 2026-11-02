import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, Voronoi

N = 26
DEADLINE = 145.0  # safety margin before 150s kill


def _elapsed(start):
    return time.time() - start


def _remaining(start):
    return DEADLINE - _elapsed(start)


def _softplus(x, k=50.0):
    """Smooth approximation of max(0, x) using softplus."""
    return np.log1p(np.exp(k * x)) / k


def _validate(p):
    """Check if configuration is valid: all circles inside [0,1]^2 and non-overlapping."""
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    # Wall check
    if np.any(x < -1e-9) or np.any(x > 1 + 1e-9):
        return False
    if np.any(y < -1e-9) or np.any(y > 1 + 1e-9):
        return False
    if np.any(r < 0):
        return False
    if np.any(r - x < -1e-9) or np.any(r - (1 - x) < -1e-9):
        return False
    if np.any(r - y < -1e-9) or np.any(r - (1 - y) < -1e-9):
        return False
    # Pairwise check
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    if np.any((r[iu[0]] + r[iu[1]]) - dist > 1e-9):
        return False
    return True


def _fix_boundaries(p):
    """Project positions and radii to satisfy wall constraints."""
    p = p.copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    return p


def _get_contact_edges(p):
    """
    Get Delaunay edges as candidate contact pairs.
    Returns array of shape (M, 2) with pairs of circle indices.
    """
    pts = p[:, :2]
    try:
        tri = Delaunay(pts)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        if not edges:
            # Fallback: all pairs
            iu = np.triu_indices(N, k=1)
            return np.column_stack([iu[0], iu[1]])
        return np.array(sorted(edges), dtype=int)
    except Exception:
        iu = np.triu_indices(N, k=1)
        return np.column_stack([iu[0], iu[1]])


def _objective_penalty(params, penalty, k):
    """
    Penalty objective using Delaunay contact graph.
    Only penalizes pairs that are Delaunay neighbors (close together).
    Uses softplus for smoothness.
    """
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]

    score = -np.sum(r)

    # Wall violations (always check all)
    wx = np.maximum(0.0, r - x) + np.maximum(0.0, r - (1.0 - x))
    wy = np.maximum(0.0, r - y) + np.maximum(0.0, r - (1.0 - y))
    wall = np.sum(wx) + np.sum(wy)

    # Get Delaunay edges for pairwise constraints
    edges = _get_contact_edges(p)
    if len(edges) > 0:
        i_arr = edges[:, 0]
        j_arr = edges[:, 1]
        dx = x[i_arr] - x[j_arr]
        dy = y[i_arr] - y[j_arr]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (r[i_arr] + r[j_arr]) - dist
        pair = np.sum(np.maximum(0.0, overlap))
    else:
        pair = 0.0

    return score + penalty * (pair + wall)


def _objective_delaunay(params, penalty, k):
    """
    Objective using softplus for smoother optimization.
    Uses Delaunay contact graph.
    """
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]

    score = -np.sum(r)

    # Wall violations
    wx = _softplus(r - x, k) + _softplus(r - (1.0 - x), k)
    wy = _softplus(r - y, k) + _softplus(r - (1.0 - y), k)
    wall = np.sum(wx) + np.sum(wy)

    # Delaunay edges
    edges = _get_contact_edges(p)
    if len(edges) > 0:
        i_arr = edges[:, 0]
        j_arr = edges[:, 1]
        dx = x[i_arr] - x[j_arr]
        dy = y[i_arr] - y[j_arr]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (r[i_arr] + r[j_arr]) - dist
        pair = np.sum(_softplus(overlap, k))
    else:
        pair = 0.0

    return score + penalty * (pair + wall)


def _initial_hex(seed):
    """Hexagonal grid initialization, selecting N closest to center."""
    rng = np.random.RandomState(seed)
    points = []
    for i in range(8):
        for j in range(8):
            x = i
            y = j * 1.5 + (0.5 if i % 2 else 0)
            points.append((x, y))
    points = np.array(points, dtype=float)
    points -= np.mean(points, axis=0)
    max_coord = np.max(np.abs(points))
    scale = 0.42 / max_coord
    points *= scale
    points += 0.5

    dists = np.sum((points - 0.5) ** 2, axis=1)
    indices = np.argsort(dists)[:N]
    p = points[indices].copy()

    # Estimate radii from nearest neighbor distances
    radii = np.zeros(N)
    for i in range(N):
        d = np.linalg.norm(p - p[i], axis=1)
        d[i] = np.inf
        nn_dist = np.min(d)
        wall_dist = np.min([p[i, 0], 1 - p[i, 0], p[i, 1], 1 - p[i, 1]])
        radii[i] = 0.5 * np.min([nn_dist, wall_dist]) * 0.9

    params = np.empty(3 * N)
    params[0::3] = p[:, 0]
    params[1::3] = p[:, 1]
    params[2::3] = radii
    params += rng.uniform(-1e-5, 1e-5, size=params.size)
    return params


def _initial_grid(seed):
    """Grid initialization."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.45 / side))
    cells = cells[:N]
    arr = np.array(cells, dtype=float).reshape(-1)
    arr += rng.uniform(-1e-5, 1e-5, size=arr.size)
    return arr


def _initial_corner_weighted(seed):
    """
    Initialization that places some circles near corners/edges
    where they can be larger, and others in the interior.
    """
    rng = np.random.RandomState(seed)
    # 4 corners, 4 edge-midpoints, 4 inner-ring, rest center
    pts = []
    # Corners
    for cx, cy in [(0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)]:
        pts.append((cx + rng.uniform(-0.02, 0.02),
                    cy + rng.uniform(-0.02, 0.02), 0.08))
    # Edge midpoints
    for cx, cy in [(0.5, 0.08), (0.5, 0.92), (0.08, 0.5), (0.92, 0.5)]:
        pts.append((cx + rng.uniform(-0.02, 0.02),
                    cy + rng.uniform(-0.02, 0.02), 0.06))
    # Inner ring
    for angle in [0, 90, 180, 270]:
        rad = np.radians(angle)
        cx = 0.5 + 0.22 * np.cos(rad)
        cy = 0.5 + 0.22 * np.sin(rad)
        pts.append((cx + rng.uniform(-0.02, 0.02),
                    cy + rng.uniform(-0.02, 0.02), 0.05))
    # Remaining in center region
    remaining = N - len(pts)
    for _ in range(remaining):
        cx = 0.5 + rng.uniform(-0.18, 0.18)
        cy = 0.5 + rng.uniform(-0.18, 0.18)
        pts.append((cx, cy, 0.04))

    p = np.array(pts, dtype=float)
    # Enforce basic validity
    p[:, 0] = np.clip(p[:, 0], 0.05, 0.95)
    p[:, 1] = np.clip(p[:, 1], 0.05, 0.95)
    wall = np.minimum(np.minimum(p[:, 0], 1 - p[:, 0]),
                      np.minimum(p[:, 1], 1 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], wall * 0.95)

    params = p.flatten()
    return params


def _enforce_validity(params):
    """Enforce all constraints by shrinking radii."""
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    for _ in range(200):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        max_ov = 0
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            max_ov = max(max_ov, ov)
            half = ov / 2.0
            shrink_i = min(half, max(p[i, 2] - 1e-6, 0))
            shrink_j = min(half, max(p[j, 2] - 1e-6, 0))
            p[i, 2] -= shrink_i
            p[j, 2] -= shrink_j
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    return p


def _score_valid(p):
    return np.sum(p[:, 2])


def _optimize_single(x0, start_time, max_time=30.0):
    """
    Run adaptive penalty annealing optimization from a starting point.
    Uses Delaunay contact graph for constraint evaluation.
    """
    x0 = _fix_boundaries(x0.reshape(N, 3)).flatten()
    best_p = x0.reshape(N, 3).copy()
    best_score = _score_valid(best_p)

    # Adaptive penalty schedule: start high, anneal down
    # Phase 1: high penalty to fix violations
    # Phase 2: moderate penalty to improve
    # Phase 3: low penalty with softplus for fine tuning

    phases = [
        {'penalty': 500.0, 'k': 100.0, 'frac': 0.3, 'method': 'L-BFGS-B',
         'maxiter': 200, 'ftol': 1e-10},
        {'penalty': 200.0, 'k': 50.0, 'frac': 0.3, 'method': 'L-BFGS-B',
         'maxiter': 300, 'ftol': 1e-12},
        {'penalty': 50.0, 'k': 20.0, 'frac': 0.25, 'method': 'L-BFGS-B',
         'maxiter': 400, 'ftol': 1e-12},
        {'penalty': 10.0, 'k': 10.0, 'frac': 0.15, 'method': 'L-BFGS-B',
         'maxiter': 500, 'ftol': 1e-14},
    ]

    lo = np.zeros(3 * N)
    hi = np.zeros(3 * N)
    for i in range(N):
        hi[3 * i] = 1.0
        hi[3 * i + 1] = 1.0
        lo[3 * i + 2] = 1e-6
        hi[3 * i + 2] = 0.5
    bounds_list = list(zip(lo, hi))

    current = x0.copy()

    for phase in phases:
        if _elapsed(start_time) > DEADLINE - 5:
            break

        penalty = phase['penalty']
        k = phase['k']
        maxiter = phase['maxiter']
        ftol = phase['ftol']

        try:
            res = minimize(
                _objective_penalty,
                current,
                args=(penalty, k),
                method=phase['method'],
                bounds=bounds_list,
                options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-6}
            )

            p_new = res.x.reshape(N, 3)
            p_new = _enforce_validity(p_new)
            s_new = _score_valid(p_new)

            if s_new > best_score:
                best_score = s_new
                best_p = p_new.copy()
                current = p_new.flatten()

            # Also try softplus version for smoother landscape
            if _elapsed(start_time) < DEADLINE - 10:
                try:
                    res2 = minimize(
                        _objective_delaunay,
                        current,
                        args=(penalty * 0.5, k),
                        method='L-BFGS-B',
                        bounds=bounds_list,
                        options={'maxiter': maxiter // 2, 'ftol': ftol, 'gtol': 1e-6}
                    )
                    p_new2 = res2.x.reshape(N, 3)
                    p_new2 = _enforce_validity(p_new2)
                    s_new2 = _score_valid(p_new2)
                    if s_new2 > best_score:
                        best_score = s_new2
                        best_p = p_new2.copy()
                        current = p_new2.flatten()
                except Exception:
                    pass

        except Exception:
            pass

    return best_p


def _shake_and_refine(p, start_time, time_limit=10.0, seed=42):
    """Stochastic local search to escape local minima."""
    best_p = p.copy()
    best_score = _score_valid(best_p)
    rng = np.random.RandomState(seed)
    sigma = 0.015

    while _elapsed(start_time) < DEADLINE - 3 and time_limit > 0:
        p_new = best_p.copy()
        p_new[:, :2] += rng.normal(0, sigma, size=(N, 2))
        p_new[:, 2] += rng.normal(0, sigma * 0.3, size=N)
        p_new = _enforce_validity(p_new)
        s_new = _score_valid(p_new)
        if s_new > best_score:
            best_score = s_new
            best_p = p_new.copy()
            time_limit -= 0.1  # rough time tracking
        else:
            time_limit -= 0.05

    return best_p


def construct():
    start_time = time.time()

    best_params = None
    best_score = -np.inf

    # Multi-start with diverse initializations
    initializers = [
        lambda s: _initial_hex(s),
        lambda s: _initial_grid(s),
        lambda s: _initial_corner_weighted(s),
    ]

    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678,
             11, 22, 33, 44, 55, 66, 77, 88, 99, 100,
             101, 202, 303, 404, 505]

    for seed in seeds:
        if _elapsed(start_time) > DEADLINE - 8:
            break
        for init_fn in initializers:
            if _elapsed(start_time) > DEADLINE - 8:
                break

            x0 = init_fn(seed)
            x0 = _enforce_validity(x0).flatten()

            # Time budget per optimization run
            time_left = DEADLINE - _elapsed(start_time) - 5
            if time_left < 3:
                break

            # Allocate time: leave room for more starts
            alloc = min(25.0, time_left / 3.0)

            try:
                p_opt = _optimize_single(x0, start_time, max_time=alloc)
                s = _score_valid(p_opt)
                if s > best_score:
                    best_score = s
                    best_params = p_opt.copy()
            except Exception:
                continue

    if best_params is None:
        best_params = _enforce_validity(_initial_grid(42))

    # Final refinement with shake
    time_left = DEADLINE - _elapsed(start_time) - 2
    if time_left > 5:
        for r_seed in [1, 2, 3, 4, 5, 6, 7, 8]:
            if _elapsed(start_time) > DEADLINE - 2:
                break
            t_limit = (DEADLINE - _elapsed(start_time) - 2) / (9 - r_seed)
            if t_limit < 1.0:
                break
            p_ref = _shake_and_refine(best_params, start_time, time_limit=t_limit, seed=r_seed)
            s_ref = _score_valid(p_ref)
            if s_ref > best_score:
                best_score = s_ref
                best_params = p_ref.copy()

    # Final validity enforcement
    best_params = _enforce_validity(best_params)

    return best_params
