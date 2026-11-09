import time
import numpy as np
from scipy.optimize import minimize

N = 21
_DEADLINE = time.time() + 140.0  # safety margin under 150s kill


def _validate(arr):
    if arr is None or arr.shape != (N, 3):
        return None
    if not np.all(np.isfinite(arr)):
        return None
    w, h, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r < 0):
        return None
    if np.any(w < 0) or np.any(w > 1) or np.any(h < 0) or np.any(h > 1):
        return None
    # non-overlap: dist >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(w[i] - w[j], h[i] - h[j])
            if d < r[i] + r[j] - 1e-7:
                return None
    return arr


def _rect_dims(a):
    # perimeter 4 => 2(w+h)=4 => w+h=2 ; aspect a = w/h
    h = 2.0 / (a + 1.0)
    w = 2.0 - h
    return w, h


def _solve_aspect(a, rng, time_left_fn):
    w, h = _rect_dims(a)
    # decision variables: 2N positions + N radii
    n = N
    x0 = np.empty(3 * n)
    for k in range(n):
        x0[2 * k] = rng.uniform(0.05, 0.95) * w
        x0[2 * k + 1] = rng.uniform(0.05, 0.95) * h
        x0[2 * n + k] = rng.uniform(0.02, 0.12)
    x0[2 * n] = 0.05  # floor-ish start

    def objective(x):
        r = x[2 * n:]
        return -np.sum(r)

    def jac(x):
        g = np.zeros(3 * n)
        g[2 * n:] = -1.0
        return g

    def con_overlap(x):
        # g = (r_i + r_j)^2 - dist^2  <= 0  <=> dist^2 - (r_i+r_j)^2 >= 0
        pos = x[:2 * n].reshape(n, 2)
        r = x[2 * n:]
        vals = []
        for i in range(n):
            for j in range(i + 1, n):
                d2 = (pos[i, 0] - pos[j, 0]) ** 2 + (pos[i, 1] - pos[j, 1]) ** 2
                s = r[i] + r[j]
                vals.append(d2 - s * s)
        return np.array(vals)

    def con_left(x):
        return x[:2 * n:2] - x[2 * n:]

    def con_right(x):
        return (w - x[:2 * n:2]) - x[2 * n:]

    def con_bottom(x):
        return x[1:2 * n:2] - x[2 * n:]

    def con_top(x):
        return (h - x[1:2 * n:2]) - x[2 * n:]

    cons = [
        {"type": "ineq", "fun": con_overlap},
        {"type": "ineq", "fun": con_left},
        {"type": "ineq", "fun": con_right},
        {"type": "ineq", "fun": con_bottom},
        {"type": "ineq", "fun": con_top},
    ]
    bounds = [(0.0, w)] * n + [(0.0, h)] * n + [(1e-4, 1.0)] * n

    res = minimize(objective, x0, jac=jac, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-10})
    if time_left_fn() < 0:
        return None
    if res.success or res.fun < 0:
        x = res.x
        arr = np.zeros((n, 3))
        arr[:, 0] = x[:2 * n:2]
        arr[:, 1] = x[1:2 * n:2]
        arr[:, 2] = x[2 * n:]
        v = _validate(arr)
        if v is not None:
            return v
    return None


def _seed_grid(a, rng):
    # a simple grid-like feasible-ish start for the aspect
    w, h = _rect_dims(a)
    cols = int(np.ceil(np.sqrt(N * w / h)))
    rows = int(np.ceil(N / cols))
    xs = np.linspace(0.1, 0.9, cols) * w
    ys = np.linspace(0.1, 0.9, rows) * h
    x0 = np.empty(3 * N)
    for k in range(N):
        i = k % cols
        j = k // cols
        x0[2 * k] = xs[i]
        x0[2 * k + 1] = ys[j]
        x0[2 * N + k] = 0.05
    return x0


def construct():
    t_start = time.time()
    rng = np.random.default_rng(12345)
    best = None
    best_val = -np.inf

    aspects = [1.0, 1.3, 0.77, 1.6, 0.62, 1.15, 0.87]
    starts_per_aspect = 3

    for a in aspects:
        for s in range(starts_per_aspect):
            if time.time() > _DEADLINE:
                break
            # mix: one near-grid start, rest random
            if s == 0:
                x0 = _seed_grid(a, rng)
            else:
                x0 = None
            # run solve (internally handles random x0 if x0 is None by reusing logic)
            if x0 is None:
                # build random x0 here
                w, h = _rect_dims(a)
                x0 = np.empty(3 * N)
                for k in range(N):
                    x0[2 * k] = rng.uniform(0.05, 0.95) * w
                    x0[2 * k + 1] = rng.uniform(0.05, 0.95) * h
                    x0[2 * N + k] = rng.uniform(0.02, 0.12)
            arr = _solve_aspect_x0(a, x0, rng)
            if arr is not None:
                val = np.sum(arr[:, 2])
                if val > best_val:
                    best_val = val
                    best = arr
        if time.time() > _DEADLINE:
            break

    if best is None:
        # fallback: seed-like uniform grid (always feasible)
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    return best


def _solve_aspect_x0(a, x0, rng):
    w, h = _rect_dims(a)
    n = N

    def objective(x):
        return -np.sum(x[2 * n:])

    def jac(x):
        g = np.zeros(3 * n)
        g[2 * n:] = -1.0
        return g

    def con_overlap(x):
        pos = x[:2 * n].reshape(n, 2)
        r = x[2 * n:]
        vals = np.empty(n * (n - 1) // 2)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                d2 = (pos[i, 0] - pos[j, 0]) ** 2 + (pos[i, 1] - pos[j, 1]) ** 2
                s = r[i] + r[j]
                vals[idx] = d2 - s * s
                idx += 1
        return vals

    def con_left(x):
        return x[:2 * n:2] - x[2 * n:]

    def con_right(x):
        return (w - x[:2 * n:2]) - x[2 * n:]

    def con_bottom(x):
        return x[1:2 * n:2] - x[2 * n:]

    def con_top(x):
        return (h - x[1:2 * n:2]) - x[2 * n:]

    cons = [
        {"type": "ineq", "fun": con_overlap},
        {"type": "ineq", "fun": con_left},
        {"type": "ineq", "fun": con_right},
        {"type": "ineq", "fun": con_bottom},
        {"type": "ineq", "fun": con_top},
    ]
    bounds = [(0.0, w)] * n + [(0.0, h)] * n + [(1e-4, 1.0)] * n

    def time_left_fn():
        return _DEADLINE - time.time()

    if time_left_fn() < 2:
        return None
    try:
        res = minimize(objective, x0, jac=jac, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 250, "ftol": 1e-11})
    except Exception:
        return None
    if time_left_fn() < 0:
        return None
    if res.fun < 0:
        x = res.x
        arr = np.zeros((n, 3))
        arr[:, 0] = x[:2 * n:2]
        arr[:, 1] = x[1:2 * n:2]
        arr[:, 2] = x[2 * n:]
        v = _validate(arr)
        if v is not None:
            return v
    return None
