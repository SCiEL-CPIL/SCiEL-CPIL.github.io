"""Circle packing in unit square: maximize sum of 26 radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
_DEADLINE = 150.0
_EPS = 1e-9


def _penalty(xy, r):
    """Penalty for overlaps and boundary violations (0 = valid)."""
    x, y = xy[:, 0], xy[:, 1]
    n = len(r)
    # boundary: circle must fit in [0,1]^2
    b = np.maximum(0.0, np.maximum(r - x, r - y))
    b = np.maximum(b, np.maximum(r - (1.0 - x), r - (1.0 - y)))
    # pairwise overlap
    d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    s = r[:, None] + r[None, :]
    o = np.maximum(0.0, s - d)
    np.fill_diagonal(o, 0.0)
    return b.sum() + o.sum()


def _objective(v):
    """Sum of squared violations (positions + radii) to minimize."""
    n = N
    xy = v[: 2 * n].reshape(n, 2)
    r = v[2 * n:]
    # clip radii to feasible range for the penalty to stay well-behaved
    r = np.clip(r, 0.0, 0.5)
    return _penalty(xy, r) ** 2


def _make_valid(xy, r):
    """Shrink radii until no overlap / boundary violation. Returns valid (xy, r)."""
    r = np.clip(r, 0.0, 0.5).copy()
    for _ in range(400):
        x, y = xy[:, 0], xy[:, 1]
        # boundary limit per circle
        lim = np.minimum(np.minimum(x, y), np.minimum(1.0 - x, 1.0 - y))
        lim = np.maximum(lim, 0.0)
        # pairwise limit
        d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
        d = np.where(np.eye(N, dtype=bool), np.inf, d)
        p_lim = np.minimum(r[:, None] + r[None, :], d)
        p_lim = np.maximum(p_lim, 0.0)
        # max radius for circle i given others
        maxr = np.minimum(lim, p_lim.min(axis=1))
        # if a circle exceeds its limit, shrink it
        if np.any(r > maxr + _EPS):
            bad = np.where(r > maxr + _EPS)[0]
            # shrink to a fraction of the limiting value
            for i in bad:
                r[i] = max(0.0, maxr[i] * 0.999)
        else:
            break
    # final safety: enforce hard limits
    x, y = xy[:, 0], xy[:, 1]
    lim = np.minimum(np.minimum(x, y), np.minimum(1.0 - x, 1.0 - y))
    r = np.minimum(r, np.maximum(lim, 0.0))
    return xy, r


def _maximize_radii(xy, r, iters=60):
    """Grow radii toward the binding constraint until valid."""
    r = r.copy()
    for _ in range(iters):
        x, y = xy[:, 0], xy[:, 1]
        lim = np.minimum(np.minimum(x, y), np.minimum(1.0 - x, 1.0 - y))
        lim = np.maximum(lim, 0.0)
        d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
        d = np.where(np.eye(N, dtype=bool), np.inf, d)
        p_lim = np.minimum(r[:, None] + r[None, :], d)
        p_lim = np.maximum(p_lim, 0.0)
        maxr = np.minimum(lim, p_lim.min(axis=1))
        # grow each circle toward its limit
        new_r = np.minimum(maxr, np.maximum(r, 0.0))
        # grow by a step; if a circle is at its limit, it stays
        r_new = 0.5 * (r + maxr)
        r_new = np.minimum(r_new, maxr)
        # keep non-decreasing
        r_new = np.maximum(r_new, r)
        if np.allclose(r_new, r, atol=1e-10):
            r = r_new
            break
        r = r_new
    # enforce validity
    return _make_valid(xy, r)


def _grid_start(jitter=0.0):
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
    pts = np.array(cells[:N])
    if jitter > 0:
        pts = pts + jitter * (np.random.rand(N, 2) - 0.5)
    pts = np.clip(pts, 0.05, 0.95)
    return pts


def _optimize_start(pts, r0, time_budget):
    """Run L-BFGS-B from a start; return improved (xy, r) or None if timed out."""
    v0 = np.concatenate([pts.reshape(-1), r0])
    start = time.time()
    try:
        res = minimize(
            _objective,
            v0,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N,
            options={"maxiter": 2000, "maxfun": 200000, "disp": False},
        )
    except Exception:
        return None
    if time.time() - start > time_budget:
        return None
    v = np.clip(res.x, 0.0, 1.0)
    v[2 * N:] = np.clip(v[2 * N:], 0.0, 0.5)
    xy = v[: 2 * N].reshape(N, 2)
    r = v[2 * N:]
    # repair + grow
    xy, r = _make_valid(xy, r)
    xy, r = _maximize_radii(xy, r)
    return xy, r


def construct():
    t_start = time.time()
    deadline = t_start + _DEADLINE - 5.0  # safety margin

    # initial equal-circle radius estimate
    side = int(np.ceil(np.sqrt(N)))
    r0 = np.full(N, 0.5 / side)

    best = None
    best_val = -1.0

    def consider(xy, r):
        nonlocal best, best_val
        xy, r = _make_valid(xy, r)
        val = r.sum()
        if val > best_val:
            best_val = val
            best = (xy.copy(), r.copy())

    # multi-start
    rng = np.random.default_rng(0)
    starts = [_grid_start(0.0)]
    for _ in range(8):
        starts.append(_grid_start(jitter=0.25))

    for k, pts in enumerate(starts):
        if time.time() > deadline:
            break
        r0k = r0.copy()
        if k > 0:
            r0k = r0 * (0.8 + 0.4 * rng.random())
        time_budget = 12.0
        out = _optimize_start(pts, r0k, time_budget)
        if out is not None:
            consider(*out)
        # also consider the repaired start itself
        consider(pts, r0k)

    if best is None:
        # fallback: trivial grid
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        arr = np.array(cells[:N])
        return arr

    xy, r = best
    # final grow pass
    xy, r = _maximize_radii(xy, r, iters=120)
    consider(xy, r)
    xy, r = best
    out = np.column_stack([xy[:, 0], xy[:, 1], r])
    return out
