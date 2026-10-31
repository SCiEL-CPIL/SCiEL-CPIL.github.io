"""Circle packing in unit square: maximize sum of radii of N disjoint circles."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26

_DEADLINE = time.time() + 150.0  # wall-clock guard; return best before this.


def _softmin_penalty(v, k):
    """Smooth approximation of max(0, -min(v)) i.e. penalty when any constraint violated.
    v: array of constraint margins (>=0 means satisfied). We want min(v) >= 0.
    Penalty = softplus(-min(v)) approximated via logsumexp."""
    # min(v) >= 0 desired. Use -min(v) = max(-v). softplus(max(-v)).
    # max(-v) via logsumexp(-v)/k approx.
    m = np.max(-v)
    if m == -np.inf:
        return 0.0
    z = -v - m
    s = np.sum(np.exp(z))
    maxval = m + np.log(s) / k
    # softplus(maxval)
    if maxval > 30:
        return maxval
    if maxval < -30:
        return 0.0
    return np.log1p(np.exp(maxval))


def _constraints(v):
    """Return array of margins (>=0 satisfied). v = [x1,y1,r1, x2,y2,r2, ...]"""
    n = len(v) // 3
    pts = v.reshape(n, 3)
    x = pts[:, 0]
    y = pts[:, 1]
    r = pts[:, 2]
    margins = []
    # inside square
    margins.append(x - r)
    margins.append(y - r)
    margins.append((1.0 - x) - r)
    margins.append((1.0 - y) - r)
    # pairwise non-overlap: dist - (ri + rj)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            margins.append(d - (r[i] + r[j]))
    return np.concatenate(margins)


def _objective_and_grad(v, k):
    """Objective to minimize: -sum(r) + lambda * penalty. Returns (f, g)."""
    n = len(v) // 3
    pts = v.reshape(n, 3)
    r = pts[:, 2]
    obj = -np.sum(r)

    # margins
    x = pts[:, 0]
    y = pts[:, 1]
    margins = []
    # inside
    margins.append(x - r)
    margins.append(y - r)
    margins.append((1.0 - x) - r)
    margins.append((1.0 - y) - r)
    # pairwise
    dmat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dmat[i, j] = np.hypot(x[i] - x[j], y[i] - y[j])
    for i in range(n):
        for j in range(i + 1, n):
            margins.append(dmat[i, j] - (r[i] + r[j]))
    vmar = np.concatenate(margins)

    # penalty = softplus(max(-vmar)) with k
    neg = -vmar
    m = np.max(neg)
    if m == -np.inf:
        pen = 0.0
        pen_grad = None
    else:
        z = neg - m
        ez = np.exp(z)
        s = np.sum(ez)
        maxval = m + np.log(s) / k
        if maxval > 30:
            pen = maxval
            pexp = 1.0
        elif maxval < -30:
            pen = 0.0
            pexp = 0.0
        else:
            pen = np.log1p(np.exp(maxval))
            pexp = np.exp(maxval) / (1.0 + np.exp(maxval))  # sigmoid(maxval)
        # d pen / d maxval = pexp
        # d maxval / d neg_l = (1/k) * ez_l / s
        w = (1.0 / k) * (ez / s)  # weight per constraint margin (on -vmar)
        # d pen / d vmar_l = -pexp * w_l
        dv = -pexp * w  # gradient of pen w.r.t. vmar

    lam = 50.0
    # Build gradient w.r.t. v
    g = np.zeros_like(v)
    # objective gradient: -sum(r) -> dr = -1
    for i in range(n):
        g[3 * i + 2] += -1.0

    if dv is not None:
        # add lam * d pen / d v
        idx = 0
        # inside: 4 groups of n
        for a in range(4):
            block = dv[idx:idx + n]
            idx += n
            if a == 0:  # x - r
                for i in range(n):
                    g[3 * i + 0] += lam * block[i]
                    g[3 * i + 2] += lam * (-block[i])
            elif a == 1:  # y - r
                for i in range(n):
                    g[3 * i + 1] += lam * block[i]
                    g[3 * i + 2] += lam * (-block[i])
            elif a == 2:  # (1-x) - r
                for i in range(n):
                    g[3 * i + 0] += lam * (-block[i])
                    g[3 * i + 2] += lam * (-block[i])
            else:  # (1-y) - r
                for i in range(n):
                    g[3 * i + 1] += lam * (-block[i])
                    g[3 * i + 2] += lam * (-block[i])
        # pairwise
        for i in range(n):
            for j in range(i + 1, n):
                c = dv[idx]
                idx += 1
                dx = (x[i] - x[j])
                dy = (y[i] - y[j])
                d = dmat[i, j]
                if d > 1e-12:
                    g[3 * i + 0] += lam * c * (dx / d)
                    g[3 * i + 1] += lam * c * (dy / d)
                    g[3 * j + 0] += lam * c * (-dx / d)
                    g[3 * j + 1] += lam * c * (-dy / d)
                g[3 * i + 2] += lam * c * (-1.0)
                g[3 * j + 2] += lam * c * (-1.0)

    f = obj + lam * pen
    return f, g


def _is_valid(pts, tol=1e-9):
    n = len(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    r = pts[:, 2]
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(y - r < -tol) or np.any(1 - x - r < -tol) or np.any(1 - y - r < -tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - tol:
                return False
    return True


def _repair(pts):
    """Clip radii to fit inside square and resolve overlaps by shrinking."""
    x = pts[:, 0].copy()
    y = pts[:, 1].copy()
    r = pts[:, 2].copy()
    # clip positions to [0,1]
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    # radius limited by distance to walls
    for i in range(len(pts)):
        r[i] = min(r[i], x[i], y[i], 1 - x[i], 1 - y[i])
    r = np.maximum(r, 0.0)
    # resolve overlaps by shrinking smaller radius
    for _ in range(200):
        changed = False
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                need = r[i] + r[j]
                if d < need:
                    # shrink proportionally
                    deficit = need - d
                    # shrink both by half deficit
                    hi = 0.5 * deficit
                    ri = r[i]
                    rj = r[j]
                    # keep proportional
                    if ri + rj <= 0:
                        continue
                    # reduce each
                    cut = deficit
                    # distribute by current radii
                    wi = ri / (ri + rj)
                    wj = rj / (ri + rj)
                    r[i] = max(0.0, ri - wi * cut)
                    r[j] = max(0.0, rj - wj * cut)
                    # re-clip to walls
                    r[i] = min(r[i], x[i], y[i], 1 - x[i], 1 - y[i])
                    r[j] = min(r[j], x[j], y[j], 1 - x[j], 1 - y[j])
                    changed = True
        if not changed:
            break
    return np.column_stack([x, y, r])


def _grid_start(seed):
    """Structured start: grid of equal circles."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side))
    cells = cells[:N]
    pts = np.array(cells, dtype=float)
    # small jitter
    pts[:, 0] += rng.uniform(-0.01, 0.01, N)
    pts[:, 1] += rng.uniform(-0.01, 0.01, N)
    pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0)
    return pts


def _random_start(seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = np.full(N, 0.05)
    return np.column_stack([x, y, r])


def construct():
    global _DEADLINE
    best_pts = None
    best_val = -np.inf

    starts = [_grid_start(0), _random_start(1), _random_start(2), _grid_start(3)]

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))   # x
        bounds.append((0.0, 1.0))   # y
        bounds.append((0.0, 0.5))   # r

    for s_idx, start in enumerate(starts):
        if time.time() > _DEADLINE - 30.0:
            break
        v0 = start.flatten()
        # anneal k from large to small (coarse to fine)
        k = 8.0
        v = v0.copy()
        for stage in range(4):
            if time.time() > _DEADLINE - 15.0:
                break
            res = minimize(
                _objective_and_grad, v, args=(k,),
                jac=True, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000}
            )
            v = res.x
            k = max(2.0, k * 0.5)
        # final high-precision polish
        if time.time() > _DEADLINE - 5.0:
            break
        res = minimize(
            _objective_and_grad, v, args=(1.0,),
            jac=True, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 5000}
        )
        v = res.x
        pts = v.reshape(N, 3)
        # repair to guarantee validity
        pts = _repair(pts)
        if _is_valid(pts):
            val = np.sum(pts[:, 2])
            if val > best_val:
                best_val = val
                best_pts = pts.copy()

    if best_pts is None:
        # fallback: valid grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_pts
