import numpy as np
import time

N = 21

def _force_directed(x, y, r, W, H, n_iter=400, dt=0.02, damping=0.9,
                    repulse=0.5, bound_stiff=2.0, time_budget=None):
    """Grow circles by pushing them apart (repulsion) while keeping them in [0,W]x[0,H]."""
    x = x.astype(float).copy()
    y = y.astype(float).copy()
    r = r.astype(float).copy()
    for it in range(n_iter):
        if time_budget is not None and time.time() > time_budget:
            break
        fx = np.zeros(N)
        fy = np.zeros(N)
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                if d < 1e-12:
                    dx = np.random.uniform(-1e-3, 1e-3)
                    dy = np.random.uniform(-1e-3, 1e-3)
                    d = np.hypot(dx, dy)
                pen = (r[i] + r[j]) - d
                if pen > 0:
                    ux = dx / d
                    uy = dy / d
                    f = repulse * pen
                    fx[i] += f * ux
                    fy[i] += f * uy
                    fx[j] -= f * ux
                    fy[j] -= f * uy
        # boundary repulsion (push inward when a circle would cross the wall)
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += bound_stiff * (r[i] - x[i])
            if x[i] > W - r[i]:
                fx[i] -= bound_stiff * (x[i] - (W - r[i]))
            if y[i] < r[i]:
                fy[i] += bound_stiff * (r[i] - y[i])
            if y[i] > H - r[i]:
                fy[i] -= bound_stiff * (y[i] - (H - r[i]))
        # integrate positions
        x += dt * fx * damping
        y += dt * fy * damping
        # clamp positions so a circle never crosses the wall
        for i in range(N):
            x[i] = min(max(x[i], r[i]), W - r[i])
            y[i] = min(max(y[i], r[i]), H - r[i])
        # grow radii: each circle can grow until it would touch a neighbor or the wall
        for i in range(N):
            r_max = min(x[i], W - x[i], y[i], H - y[i])
            for j in range(N):
                if i == j:
                    continue
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                cand = d - r[j]
                if cand < r_max:
                    r_max = cand
            r_max = max(r_max, 1e-7)
            r[i] = max(r[i], r_max)
    return x, y, r


def _relax_radii(x, y, r, W, H, n_iter=15, tol=1e-9):
    """Fixed positions: push every radius to its maximal feasible value."""
    r = r.astype(float).copy()
    for _ in range(n_iter):
        improved = False
        for i in range(N):
            r_max = min(x[i], W - x[i], y[i], H - y[i])
            for j in range(N):
                if i == j:
                    continue
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                cand = d - r[j]
                if cand < r_max:
                    r_max = cand
            r_max = max(r_max, 1e-7)
            if r_max > r[i] + tol:
                r[i] = r_max
                improved = True
        if not improved:
            break
    return r


def _score(x, y, r, W, H, tol=1e-6):
    if W <= 0 or H <= 0 or 2.0 * (W + H) > 4.0 + tol:
        return -np.inf
    for i in range(N):
        if x[i] < r[i] - tol or x[i] > W - r[i] + tol:
            return -np.inf
        if y[i] < r[i] - tol or y[i] > H - r[i] + tol:
            return -np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
            if d2 < (r[i] + r[j]) ** 2 - tol:
                return -np.inf
    return float(np.sum(r))


def _grid_init(W, H, jitter=0.02, seed=None):
    rng = np.random.default_rng(seed)
    cols = 6
    rows = 4  # 24 cells, use 21
    xs = np.zeros(N)
    ys = np.zeros(N)
    r = np.zeros(N)
    k = 0
    for j in range(rows):
        for i in range(cols):
            if k >= N:
                break
            cx = (i + 0.5) * W / cols
            cy = (j + 0.5) * H / rows
            xs[k] = cx + rng.normal(0, jitter)
            ys[k] = cy + rng.normal(0, jitter)
            r[k] = 0.4 * min(W / cols, H / rows)
            k += 1
    return xs, ys, r


def _random_init(W, H, seed=None):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.1, W - 0.1, N)
    ys = rng.uniform(0.1, H - 0.1, N)
    r = rng.uniform(0.03, 0.12, N)
    return xs, ys, r


def construct():
    start = time.time()
    deadline = start + 140.0  # hard wall-clock guard (evaluator kills at 150s)

    best_score = -np.inf
    best = None

    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 1.33, 0.75, 1.5, 0.66]

    trial = 0
    while time.time() < deadline - 5.0:
        trial += 1
        if trial <= len(aspect_ratios):
            ratio = aspect_ratios[trial - 1]
        else:
            ratio = np.random.uniform(0.6, 1.6)
        W = 2.0 * ratio / (1.0 + ratio)
        H = 2.0 - W

        if trial % 3 == 0:
            xs, ys, r = _random_init(W, H, seed=trial)
        else:
            xs, ys, r = _grid_init(W, H, jitter=0.02, seed=trial)

        # Force-directed growth with a time budget for this trial
        trial_budget = min(deadline - 6.0, time.time() + 12.0)
        xs, ys, r = _force_directed(xs, ys, r, W, H,
                                    n_iter=500, dt=0.03,
                                    time_budget=trial_budget)
        # Polish radii at fixed positions
        r = _relax_radii(xs, ys, r, W, H, n_iter=25)
        # A short second force pass to re-settle after radius polish
        if time.time() < deadline - 8.0:
            xs, ys, r = _force_directed(xs, ys, r, W, H,
                                        n_iter=150, dt=0.02,
                                        time_budget=min(deadline - 6.0, time.time() + 5.0))
            r = _relax_radii(xs, ys, r, W, H, n_iter=25)

        s = _score(xs, ys, r, W, H)
        if s > best_score:
            best_score = s
            best = (xs, ys, r, W, H)

    if best is None:
        # Fallback: simple grid in a 1x1 square
        xs = np.zeros(N); ys = np.zeros(N); r = np.zeros(N)
        k = 0
        for j in range(4):
            for i in range(6):
                if k >= N:
                    break
                xs[k] = (i + 0.5) / 6.0
                ys[k] = (j + 0.5) / 4.0
                r[k] = 0.4 * min(1.0 / 6.0, 1.0 / 4.0)
                k += 1
        best = (xs, ys, r, 1.0, 1.0)

    xs, ys, r, W, H = best
    # Final safety: shrink radii a hair to guarantee disjoint interiors under strict check
    r = r * (1.0 - 1e-5)
    result = np.column_stack((xs, ys, r))
    return result
