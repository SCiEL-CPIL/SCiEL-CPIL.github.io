"""Circle packing in unit square: 26 disjoint circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, NonlinearConstraint

N = 26
EPS = 1e-8
DEADLINE = 145.0  # safety margin before 150s kill


def _pairwise_constraints(p, iu):
    """Return pairwise distance - (r_i + r_j) for active pairs."""
    x = p[iu[0], 0]
    y = p[iu[0], 1]
    r_i = np.maximum(p[iu[0], 2], 1e-6)
    xj = p[iu[1], 0]
    yj = p[iu[1], 1]
    r_j = np.maximum(p[iu[1], 2], 1e-6)
    dx = x - xj
    dy = y - yj
    dist = np.sqrt(dx * dx + dy * dy)
    return dist - (r_i + r_j)


def _wall_constraints(p):
    """Return wall clearance: min(x-r, 1-x-r, y-r, 1-y-r) per circle."""
    x = p[:, 0]
    y = p[:, 1]
    r = np.maximum(p[:, 2], 1e-6)
    return np.minimum(np.minimum(x - r, 1.0 - x - r),
                      np.minimum(y - r, 1.0 - y - r))


def _objective(p):
    """Minimize negative sum of radii."""
    return -np.sum(np.maximum(p[:, 2], 1e-6))


def _objective_grad(p):
    """Gradient of objective."""
    g = np.zeros_like(p)
    g[2::3] = -1.0
    return g


def _pairwise_grad(p, iu):
    """Jacobian of pairwise constraints w.r.t. p."""
    n = N
    m = len(iu[0])
    J = np.zeros((m, 3 * n))
    x = p[:, 0]
    y = p[:, 1]
    r = np.maximum(p[:, 2], 1e-6)
    
    for k in range(m):
        i = iu[0][k]
        j = iu[1][k]
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx * dx + dy * dy)
        dist_safe = max(dist, 1e-10)
        J[k, 3*i] = dx / dist_safe
        J[k, 3*i+1] = dy / dist_safe
        J[k, 3*i+2] = -1.0
        J[k, 3*j] = -dx / dist_safe
        J[k, 3*j+1] = -dy / dist_safe
        J[k, 3*j+2] = -1.0
    return J


def _wall_grad(p):
    """Jacobian of wall constraints w.r.t. p."""
    n = N
    J = np.zeros((4 * n, 3 * n))
    x = p[:, 0]
    y = p[:, 1]
    r = np.maximum(p[:, 2], 1e-6)
    
    for i in range(n):
        # x - r >= 0
        J[4*i, 3*i] = 1.0
        J[4*i, 3*i+2] = -1.0
        # 1-x-r >= 0
        J[4*i+1, 3*i] = -1.0
        J[4*i+1, 3*i+2] = -1.0
        # y - r >= 0
        J[4*i+2, 3*i+1] = 1.0
        J[4*i+2, 3*i+2] = -1.0
        # 1-y-r >= 0
        J[4*i+3, 3*i+1] = -1.0
        J[4*i+3, 3*i+2] = -1.0
    return J


def _hex_initial(seed):
    """Hexagonal grid initialization."""
    rng = np.random.RandomState(seed)
    points = []
    for i in range(7):
        for j in range(7):
            x = i
            y = j * 1.5 + (0.5 if i % 2 else 0)
            points.append((x, y))
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    max_coord = np.max(np.abs(points))
    scale = 0.45 / max_coord
    points *= scale
    points += 0.5
    
    dists = np.sum((points - 0.5)**2, axis=1)
    indices = np.argsort(dists)[:N]
    
    p = points[indices]
    radii = np.full(N, 0.05)
    
    params = np.empty(3 * N)
    params[0::3] = p[:, 0]
    params[1::3] = p[:, 1]
    params[2::3] = radii
    
    params += rng.uniform(-1e-4, 1e-4, size=params.size)
    return params


def _grid_initial(seed):
    """Grid of equal circles with small jitter."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4))
    cells = cells[:N]
    arr = np.array(cells, dtype=float).reshape(-1)
    arr += rng.uniform(-1e-5, 1e-5, size=arr.size)
    return arr


def _voronoi_initial(seed):
    """Voronoi-based initialization: place points, compute inradius of Voronoi cells."""
    rng = np.random.RandomState(seed)
    # Generate random points
    x = rng.uniform(0.1, 0.9, size=N)
    y = rng.uniform(0.1, 0.9, size=N)
    
    # Estimate radius as half the minimum distance to other points or walls
    radii = np.zeros(N)
    for i in range(N):
        min_dist = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
        for j in range(N):
            if i != j:
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                min_dist = min(min_dist, d * 0.5)
        radii[i] = min_dist * 0.95
    
    params = np.empty(3 * N)
    params[0::3] = x
    params[1::3] = y
    params[2::3] = radii
    
    return params


def _bounds():
    lo = np.zeros(3 * N)
    hi = np.zeros(3 * N)
    for i in range(N):
        lo[3*i] = 0.0
        hi[3*i] = 1.0
        lo[3*i+1] = 0.0
        hi[3*i+1] = 1.0
        lo[3*i+2] = 1e-6
        hi[3*i+2] = 0.5
    return (lo, hi)


def _enforce_validity(params):
    """Enforce all constraints: walls and pairwise non-overlap by shrinking radii."""
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    # Clamp to walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    # Iteratively resolve overlaps by shrinking the smaller circle
    for _ in range(1000):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    return p


def _score_valid(p):
    """Compute sum of radii for a valid configuration."""
    return np.sum(p[:, 2])


def _optimize_with_constraints(params, iu, deadline):
    """Optimize using SLSQP with explicit constraints."""
    lo, hi = _bounds()
    bounds_list = list(zip(lo, hi))
    
    # Nonlinear constraints: pairwise distances
    pairwise_con = NonlinearConstraint(
        lambda p: _pairwise_constraints(p.reshape(N, 3), iu),
        0.0, np.inf
    )
    
    wall_con = NonlinearConstraint(
        lambda p: _wall_constraints(p.reshape(N, 3)),
        0.0, np.inf
    )
    
    constraints = [pairwise_con, wall_con]
    
    try:
        res = minimize(
            lambda p: _objective(p.reshape(N, 3)),
            params,
            jac=lambda p: _objective_grad(p.reshape(N, 3)),
            method='SLSQP',
            bounds=bounds_list,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
        )
        if time.time() > deadline - 2:
            return params
        return res.x
    except Exception:
        return params


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_params = None
    best_score = -np.inf

    iu = np.triu_indices(N, k=1)

    # Multi-start with different initializations
    initializers = [
        lambda s: _hex_initial(s),
        lambda s: _grid_initial(s),
        lambda s: _voronoi_initial(s),
    ]
    
    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828]
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        for init_fn in initializers:
            if time.time() > deadline - 10:
                break
            params = init_fn(seed)
            
            # Phase 1: Optimize with constraints
            if time.time() < deadline - 5:
                params = _optimize_with_constraints(params, iu, deadline)
            
            # Enforce validity and score
            valid_p = _enforce_validity(params)
            score = _score_valid(valid_p)

            if score > best_score:
                best_score = score
                best_params = valid_p.copy()

    if best_params is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_params
