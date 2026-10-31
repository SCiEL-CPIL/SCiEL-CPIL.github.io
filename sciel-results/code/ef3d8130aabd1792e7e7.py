"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 26


def _compute_radii(x, y):
    """Radius of each circle = min(dist to nearest other circle/2, dist to walls)."""
    n = len(x)
    if n <= 1:
        return np.full(n, 0.5)
    tree = cKDTree(np.column_stack([x, y]))
    dist, _ = tree.query(np.column_stack([x, y]), k=2)
    nn = dist[:, 1] / 2.0
    wall = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    return np.maximum(0.0, np.minimum(nn, wall))


def _objective(x, y):
    r = _compute_radii(x, y)
    return r.sum()


def _project(x, y):
    """Keep centers inside [0,1]^2 (walls handled by radius)."""
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    return x, y


def _init_hexagonal(n, jitter=0.0, rng=None):
    """Hexagonal close-packing style initialization in unit square."""
    if rng is None:
        rng = np.random.default_rng(0)
    # Aim for roughly sqrt(n) rows
    rows = int(np.ceil(np.sqrt(n * 0.866)))
    pts = []
    for j in range(rows):
        y = (j + 0.5) / rows
        offset = (j % 2) * 0.5
        cols = int(np.ceil(np.sqrt(n) * 1.15))
        for i in range(cols):
            x = (i + 0.5 + offset) / (cols + 0.5)
            if 0 <= x <= 1 and 0 <= y <= 1:
                pts.append((x, y))
        if len(pts) >= n:
            break
    # If not enough, fill with a grid
    if len(pts) < n:
        side = int(np.ceil(np.sqrt(n)))
        for i in range(side):
            for j in range(side):
                pts.append(((i + 0.5) / side, (j + 0.5) / side))
    pts = pts[:n]
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    if jitter > 0:
        x += rng.uniform(-jitter, jitter, n)
        y += rng.uniform(-jitter, jitter, n)
    x, y = _project(x, y)
    return x, y


def _init_random(n, rng):
    x = rng.uniform(0.05, 0.95, n)
    y = rng.uniform(0.05, 0.95, n)
    return x, y


def _local_search(x, y, rng, max_iter=4000, base_step=0.02, time_budget=60.0):
    """Adaptive step-size local search (gradient-free) maximizing sum of radii."""
    x = x.copy()
    y = y.copy()
    cur = _objective(x, y)
    t0 = time.time()
    step = base_step
    n = len(x)

    for it in range(max_iter):
        if time.time() - t0 > time_budget:
            break
        # Try a random perturbation of a random subset
        k = rng.integers(1, max(2, n // 4) + 1)
        idx = rng.choice(n, size=k, replace=False)
        dx = rng.normal(0, step, k)
        dy = rng.normal(0, step, k)

        xn = x.copy()
        yn = y.copy()
        xn[idx] += dx
        yn[idx] += dy
        xn, yn = _project(xn, yn)
        new = _objective(xn, yn)
        if new > cur + 1e-12:
            x, y = xn, yn
            cur = new
            step = min(0.1, step * 1.02)
        else:
            step = max(1e-4, step * 0.97)
    return x, y, cur


def _polish(x, y, rng, max_iter=2000, base_step=0.003, time_budget=20.0):
    """High-precision polish with smaller steps."""
    return _local_search(x, y, rng, max_iter=max_iter, base_step=base_step,
                         time_budget=time_budget)


def construct():
    t_start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    rng = np.random.default_rng(2024)

    best_x, best_y = None, None
    best_val = -1.0

    # Phase 1: multiple restarts with hexagonal + jitter and random inits
    n_restarts = 8
    per_restart = 12.0
    for r in range(n_restarts):
        if time.time() - t_start > DEADLINE - 25.0:
            break
        if r % 2 == 0:
            jitter = 0.03 * (r // 2 + 1)
            x, y = _init_hexagonal(N, jitter=jitter, rng=rng)
        else:
            x, y = _init_random(N, rng)
        # Quick local search
        x, y, val = _local_search(x, y, rng, max_iter=1500, base_step=0.03,
                                  time_budget=per_restart)
        if val > best_val:
            best_val = val
            best_x, best_y = x.copy(), y.copy()

    # Phase 2: polish the best solution
    if best_x is not None:
        remaining = DEADLINE - 5.0 - (time.time() - t_start)
        if remaining > 3.0:
            best_x, best_y, best_val = _polish(
                best_x, best_y, rng,
                max_iter=3000, base_step=0.004,
                time_budget=max(3.0, remaining))

    # Fallback: ensure we always have a valid result
    if best_x is None:
        x, y = _init_hexagonal(N, jitter=0.0, rng=rng)
        best_x, best_y = x, y

    r = _compute_radii(best_x, best_y)
    # Ensure strict validity: shrink radii slightly and clamp centers
    r = np.maximum(0.0, r - 1e-9)
    best_x = np.clip(best_x, r, 1.0 - r)
    best_y = np.clip(best_y, r, 1.0 - r)
    r = np.minimum(r, np.minimum(np.minimum(best_x, 1.0 - best_x),
                                 np.minimum(best_y, 1.0 - best_y)))

    out = np.column_stack([best_x, best_y, r])
    return out
