"""Circle packing in the unit square: maximize sum of radii for 26 disjoint circles."""
import time
import numpy as np

N = 26
_T0 = time.time()
_DEADLINE = _T0 + 115.0  # safety margin under the 150 s wall-clock limit


def _objective(v, W):
    # v = [x(26), y(26), logr(26)]
    x = v[:N]
    y = v[N:2 * N]
    r = np.exp(v[2 * N:])
    # pair distances vs required (ri + rj)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    req = r[:, None] + r[None, :]
    pen = np.maximum(req - d, 0.0)
    pen = pen + pen.T - np.diag(np.diag(pen))  # count each pair once
    # square containment
    pen += np.maximum(r - x, 0.0)
    pen += np.maximum(r - y, 0.0)
    pen += np.maximum(r - (1.0 - x), 0.0)
    pen += np.maximum(r - (1.0 - y), 0.0)
    return np.sum(r) - W * np.sum(pen)


def _run(start, W, maxiter):
    from scipy.optimize import minimize
    res = minimize(_objective, start, method="L-BFGS-B",
                   args=(W,),
                   bounds=[(0.0, 1.0)] * (2 * N) + [(0.0, 0.0)] * N,
                   options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-10,
                            "maxls": 40, "disp": False})
    return res


def _feasible(x, y, r, tol=1e-7):
    if r.min() <= 0:
        return False
    if (x < -tol).any() or (y < -tol).any() or (x > 1 + tol).any() or (y > 1 + tol).any():
        return False
    if (x < r).any() or (y < r).any() or (x > 1 - r).any() or (y > 1 - r).any():
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    if (d < (r[:, None] + r[None, :]) - tol).any():
        return False
    return True


def _final_polish(x, y, r, W, budget):
    v0 = np.concatenate([x, y, np.log(r)])
    for maxiter in (2000, 4000):
        if time.time() > budget:
            break
        try:
            res = _run(v0, W, maxiter)
        except Exception:
            break
        if res.success or res.fun < _objective(v0, W):
            v0 = res.x
    x = np.clip(v0[:N], 0.0, 1.0)
    y = np.clip(v0[N:2 * N], 0.0, 1.0)
    r = np.exp(v0[2 * N:])
    # clamp radii so circles fit the square exactly
    r = np.minimum(r, np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y)))
    return x, y, r


def construct():
    rng = np.random.default_rng(20240517)
    W = 60.0
    best = None
    best_val = -np.inf

    # ---- diverse initializations ----
    def init_grid():
        side = 5
        xs = (np.arange(side) + 0.5) / side
        ys = (np.arange(side) + 0.5) / side
        gx, gy = np.meshgrid(xs, ys)
        gx = gx.ravel(); gy = gy.ravel()
        r = np.full(N, 0.5 / side * 0.9)
        return gx[:N], gy[:N], r

    def init_hex():
        xs = np.linspace(0.12, 0.88, 6)
        ys = np.linspace(0.15, 0.85, 5)
        gx, gy = np.meshgrid(xs, ys)
        gx = gx.ravel(); gy = gy.ravel()
        r = np.full(N, 0.10)
        return gx[:N], gy[:N], r

    def init_random():
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        r = rng.uniform(0.05, 0.13, N)
        return x, y, r

    def init_ring():
        # one near center, rest on a ring
        x = np.empty(N); y = np.empty(N); r = np.empty(N)
        x[0] = 0.5; y[0] = 0.5; r[0] = 0.12
        th = rng.uniform(0, 2 * np.pi, N - 1)
        rad = rng.uniform(0.30, 0.38, N - 1)
        x[1:] = np.clip(0.5 + rad * np.cos(th), 0.1, 0.9)
        y[1:] = np.clip(0.5 + rad * np.sin(th), 0.1, 0.9)
        r[1:] = rng.uniform(0.06, 0.11, N - 1)
        return x, y, r

    inits = [init_grid(), init_hex(), init_ring()]
    for _ in range(24):
        inits.append(init_random())

    for x, y, r in inits:
        if time.time() > _DEADLINE:
            break
        v0 = np.concatenate([x, y, np.log(r)])
        try:
            res = _run(v0, W, 600)
        except Exception:
            continue
        xx = np.clip(res.x[:N], 0.0, 1.0)
        yy = np.clip(res.x[N:2 * N], 0.0, 1.0)
        rr = np.exp(res.x[2 * N:])
        rr = np.minimum(rr, np.minimum(np.minimum(xx, 1.0 - xx), np.minimum(yy, 1.0 - yy)))
        if not _feasible(xx, yy, rr):
            continue
        val = float(np.sum(rr))
        if val > best_val:
            best_val = val
            best = (xx.copy(), yy.copy(), rr.copy())

    # ---- fallback: guaranteed-valid grid if nothing feasible found ----
    if best is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # ---- final high-precision polish on the best result ----
    x, y, r = best
    x, y, r = _final_polish(x, y, r, W, _DEADLINE)
    if not _feasible(x, y, r):
        x, y, r = best  # keep the previously feasible best

    return np.column_stack([x, y, r])
