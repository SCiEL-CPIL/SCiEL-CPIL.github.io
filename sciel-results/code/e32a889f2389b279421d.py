"""Pack 100 equal spheres in a unit cube; maximize common radius."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100
HALF = 0.5


def _score(c):
    """Return the true bottleneck radius for a valid centre array."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # inside the cube
    if np.any(c < -HALF) or np.any(c > HALF):
        return 0.0
    # pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    # wall clearance
    wall = np.min(HALF - np.abs(c))
    return float(min(min_pair, wall))


def _initial_grid():
    """Simple cubic-ish grid as a warm start."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - HALF
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _feasibility_penalty(c, r):
    """Penalty that is ~0 iff all pairwise dist >= 2r and wall >= r.
    Uses a sum of squared violations so it is smooth-ish for the optimizer."""
    c = np.asarray(c, dtype=np.float64)
    # pairwise
    diff = c[:, None, :] - c[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)
    viol_pair = np.maximum(0.0, 2.0 * r - d)
    # wall
    viol_wall = np.maximum(0.0, r - (HALF - np.abs(c)))
    return float(np.sum(viol_pair ** 2) + np.sum(viol_wall ** 2))


def _try_radius(r, c0, deadline, max_iter):
    """Local optimization to make radius r feasible. Return (feasible, best_c, best_pen)."""
    best_c = c0.copy()
    best_pen = _feasibility_penalty(c0, r)

    def obj(x):
        return _feasibility_penalty(x.reshape(N, 3), r)

    x0 = c0.reshape(-1).copy()
    best_x = x0.copy()

    # A few restarts from jittered copies to escape local minima
    rng = np.random.default_rng(0)
    attempts = 0
    for restart in range(4):
        if time.time() > deadline:
            break
        if restart == 0:
            x0 = c0.reshape(-1).copy()
        else:
            x0 = c0.reshape(-1).copy() + rng.normal(0, 0.02 * r, size=N * 3)
            # clip into cube
            x0 = np.clip(x0, -HALF, HALF)
        try:
            res = minimize(
                obj, x0, method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-12}
            )
        except Exception:
            continue
        attempts += 1
        pen = float(res.fun)
        if pen < best_pen:
            best_pen = pen
            best_x = res.x.copy()
            best_c = best_x.reshape(N, 3)
        if best_pen < 1e-10:
            break

    feasible = best_pen < 1e-8
    return feasible, best_c, best_pen


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    c = _initial_grid()
    r0 = _score(c)
    best_c = c.copy()
    best_r = r0

    # Binary search on radius using penalty feasibility
    lo, hi = r0, 0.12  # upper bound above best-known
    # First make sure hi is feasible from a good start by trying; if not, lower it.
    # We'll do a coarse binary search.
    for it in range(30):
        if time.time() > deadline - 20.0:
            break
        mid = 0.5 * (lo + hi)
        feasible, c_try, pen = _try_radius(mid, best_c, deadline, max_iter=4000)
        if feasible:
            lo = mid
            best_c = c_try
            # recompute true score to be safe
            true_r = _score(best_c)
            if true_r > best_r:
                best_r = true_r
        else:
            hi = mid
            # use the improved (penalty-reduced) configuration as new start
            if pen < _feasibility_penalty(best_c, mid):
                best_c = c_try

    # High-precision polish: directly maximize the true bottleneck via a
    # smooth surrogate (soft-min) using L-BFGS-B with bounds, then verify.
    if time.time() < deadline - 10.0:
        def _surrogate(x):
            cc = x.reshape(N, 3)
            diff = cc[:, None, :] - cc[None, :, :]
            d2 = np.einsum('ijk,ijk->ij', diff, diff)
            np.fill_diagonal(d2, np.inf)
            d = np.sqrt(d2)
            # soft-min of pairwise/2 and wall clearances
            vals = np.concatenate([d / 2.0, (HALF - np.abs(cc)).ravel()])
            vals = vals.ravel()
            # log-sum-exp based soft-min
            m = np.min(vals)
            s = m - np.logaddexp.reduce(vals - m)
            return -s  # maximize soft-min

        x0 = best_c.reshape(-1).copy()
        bounds = [(-HALF, HALF)] * (N * 3)
        try:
            res = minimize(
                _surrogate, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            cand = res.x.reshape(N, 3)
            r_polish = _score(cand)
            if r_polish > best_r:
                best_r = r_polish
                best_c = cand
        except Exception:
            pass

    # Final safety: ensure validity
    if _score(best_c) <= 0.0:
        # fall back to grid
        c = _initial_grid()
        return c
    return best_c


if __name__ == '__main__':
    print(_score(construct()))
