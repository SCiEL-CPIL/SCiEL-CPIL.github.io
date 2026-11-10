# EVOLVE-BLOCK-START
import time
import math
import numpy as np
import random

_T0 = time.monotonic()
_DEADLINE = 600.0

# Sofa grid points
_XI = np.arange(-3.0, 1.5, 0.01)
_YJ = np.arange(-1.5, 1.5, 0.01)
_SOFA = np.stack([np.repeat(_XI, len(_YJ)), np.tile(_YJ, len(_XI))], axis=1)
_NSOFA = _SOFA.shape[0]

# Precompute base-corridor half-extents along u for each v (in the base frame)
# Base = {0<=u<=1, -3<=v<=1} U {-3<=u<=1, 0<=v<=1}
# For a given v:
#   if -3<=v<0: only arm1 -> u in [0,1] -> umin=0, umax=1
#   if 0<=v<=1: both -> u in [-3,1] -> umin=-3, umax=1
#   if 1<v<=3: none (but sofa v is within [-1.5,1.5], so v in [1,1.5] -> none)
_VMIN, _VMAX = -3.0, 3.0
# We'll compute umin(v), umax(v) for the sofa's v range [-1.5, 1.5]
def _base_extents(v):
    # v array
    umin = np.where(v < 0.0, 0.0, -3.0)
    umax = np.where(v <= 1.0, 1.0, -np.inf)
    return umin, umax

_UMIN_V, _UMAX_V = _base_extents(_SOFA[:, 1])
_UMAX_VALID = np.isfinite(_UMAX_V)


def _eval_path(path):
    """Return the sofa area (count*1e-4) for a full path (incl endpoints)."""
    n = path.shape[0]
    if n < 2:
        return 0.0
    # Build densified list of (x,y,a)
    dens = [(0.0, 0.0, 0.0)]
    for i in range(n - 1):
        x1, y1, a1 = path[i]
        x2, y2, a2 = path[i + 1]
        m = max(1, int(math.ceil(abs(a2 - a1))))
        for k in range(1, m + 1):
            t = k / m
            dens.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1), a1 + t * (a2 - a1)))
    dens.append((-2.0, 0.0, 90.0))
    dens = np.array(dens, dtype=np.float64)

    # Fast reject: if any placement has no overlap with sofa bbox, area 0
    # Precompute sofa bbox
    sx_min, sx_max = _SOFA[:, 0].min(), _SOFA[:, 0].max()
    sy_min, sy_max = _SOFA[:, 1].min(), _SOFA[:, 1].max()

    mask = np.ones(_NSOFA, dtype=bool)
    for x, y, a in dens:
        rad = math.radians(a)
        c, s = math.cos(rad), math.sin(rad)
        # w' = R(-a)(w - (x,y)); R(-a) = [[c, s], [-s, c]]
        dx = _SOFA[:, 0] - x
        dy = _SOFA[:, 1] - y
        u = c * dx + s * dy
        v = -s * dx + c * dy
        # inside base: (0<=u<=1 & -3<=v<=1) or (-3<=u<=1 & 0<=v<=1)
        arm1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        arm2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        inside = arm1 | arm2
        mask &= inside
        if not mask.any():
            return 0.0
    return float(mask.sum()) * 1e-4


def _eval_mid(path_mid):
    """path_mid excludes the fixed endpoints."""
    full = np.vstack([[0.0, 0.0, 0.0], path_mid, [-2.0, 0.0, 90.0]])
    return _eval_path(full)


def _random_path(n):
    # angles strictly increasing from (0,90)
    inner = np.sort(np.random.uniform(2.0, 88.0, size=n))
    # x,y: wander near the inner corner region
    x = np.random.uniform(-0.8, 0.2, size=n)
    y = np.random.uniform(-0.2, 0.5, size=n)
    return np.column_stack([x, y, inner])


def _local_search(path, n_iter, step, deadline, best_val, best_path):
    cur = path.copy()
    cur_val = _eval_mid(cur)
    if cur_val > best_val:
        best_val, best_path = cur_val, cur.copy()
    n = cur.shape[0]
    for it in range(n_iter):
        if time.monotonic() > deadline:
            break
        # pick a random row and a random coordinate
        r = random.randrange(n)
        c = random.randrange(3)
        old = cur[r, c]
        if c == 2:
            # keep angle in (0,90) and roughly increasing
            new = old + random.gauss(0, step)
            new = min(89.0, max(1.0, new))
            # enforce monotonic-ish: clamp between neighbors
            if r > 0:
                new = max(new, cur[r - 1, 2] + 0.5)
            if r < n - 1:
                new = min(new, cur[r + 1, 2] - 0.5)
        else:
            new = old + random.gauss(0, step)
        cur[r, c] = new
        val = _eval_mid(cur)
        if val >= cur_val:
            cur_val = val
            if val > best_val:
                best_val, best_path = val, cur.copy()
        else:
            cur[r, c] = old
    return best_val, best_path


def moving_sofa_2d() -> np.ndarray:
    deadline = _T0 + _DEADLINE - 25.0  # safety margin
    random.seed(12345)

    # Seed with the known-good baseline path
    base = np.array(
        [
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ],
        dtype=np.float64,
    )
    best_val = _eval_mid(base)
    best_path = base.copy()

    n_restarts = 24
    n_iters = 400
    n_rows = 8

    for r in range(n_restarts):
        if time.monotonic() > deadline:
            break
        if r == 0:
            start = base.copy()
        else:
            # jitter the best path for diversity
            start = best_path + np.random.normal(0, 0.08, size=best_path.shape)
            start[:, 2] = np.sort(start[:, 2])
            start[:, 2] = np.clip(start[:, 2], 1.0, 89.0)
        # local search with a couple of step scales
        for step in (0.15, 0.05, 0.02):
            if time.monotonic() > deadline:
                break
            best_val, best_path = _local_search(
                start, n_iters, step, deadline, best_val, best_path
            )
            start = best_path.copy()

    # Final polish
    if time.monotonic() < deadline:
        best_val, best_path = _local_search(
            best_path, 600, 0.01, deadline, best_val, best_path
        )

    # Ensure a valid, finite result
    if not (np.all(np.isfinite(best_path)) and best_path.shape[0] >= 1):
        return base
    return best_path


# EVOLVE-BLOCK-END
