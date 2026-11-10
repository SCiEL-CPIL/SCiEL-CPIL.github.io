# EVOLVE-BLOCK-START
import time
import numpy as np

# ---------------------------------------------------------------------------
# Precompute the sofa grid (fixed by the task contract).
# ---------------------------------------------------------------------------
_U = np.arange(-3.0, 1.5, 0.01)
_V = np.arange(-1.5, 1.5, 0.01)
_SX, _SY = np.meshgrid(_U, _V, indexing='ij')
_SX = _SX.ravel()
_SY = _SY.ravel()
_NPTS = _SX.size

_START = (0.0, 0.0, 0.0)
_END = (-2.0, 0.0, 90.0)


def _build_placements(waypoints):
    """Return (x, y, c, s) arrays for the full placement set the evaluator uses."""
    full = [_START]
    for w in waypoints:
        full.append((float(w[0]), float(w[1]), float(w[2])))
    full.append(_END)

    xs = [full[0][0]]
    ys = [full[0][1]]
    as_ = [full[0][2]]
    for i in range(len(full) - 1):
        a1 = full[i][2]
        a2 = full[i + 1][2]
        m = max(1, int(np.ceil(abs(a2 - a1) - 1e-9)))
        for k in range(1, m + 1):
            t = k / m
            xs.append(full[i][0] + t * (full[i + 1][0] - full[i][0]))
            ys.append(full[i][1] + t * (full[i + 1][1] - full[i][1]))
            as_.append(a1 + t * (a2 - a1))

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    rad = np.deg2rad(as_)
    return x, y, np.cos(rad), np.sin(rad)


def _compute_value(waypoints):
    """Number of sofa points inside every placement (int)."""
    x, y, c, s = _build_placements(waypoints)
    n = x.size
    dx = _SX[None, :] - x[:, None]
    dy = _SY[None, :] - y[:, None]
    # local coords: R(-a) applied to (w - corner)
    u = c * dx + s * dy
    v = -s * dx + c * dy
    m1 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    m2 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
    m = m1 | m2
    if not m.all(axis=0):
        return 0
    return int(m.all(axis=1).sum())


def _perturb(way, amt):
    new = np.empty_like(way)
    for i in range(way.shape[0]):
        new[i] = way[i] + amt * np.random.randn(3)
    return new


def _optimize(seed, deadline):
    best = seed
    bv = _compute_value(best)
    rng = np.random.default_rng(12345)

    while time.time() < deadline:
        way = best
        if rng.random() < 0.25:
            way = seed.copy()
        # coordinate local search
        for _ in range(60):
            if time.time() > deadline:
                break
            cand = _perturb(way, 0.03)
            v = _compute_value(cand)
            if v > bv:
                bv = v
                way = cand
                best = cand.copy()
        # occasional larger jump
        if rng.random() < 0.3:
            cand = _perturb(best, 0.15)
            v = _compute_value(cand)
            if v > bv:
                bv = v
                best = cand.copy()
    return best


def moving_sofa_2d() -> np.ndarray:
    """Return a valid (n,3) path (n>=1, finite) that maximizes the sofa area."""
    t0 = time.time()
    deadline = t0 + 600.0 - 30.0  # 30 s safety margin

    seed = np.array(
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

    # Ensure the seed itself is valid (it is, score 1.0).
    if _compute_value(seed) == 0:
        return seed

    best = _optimize(seed, deadline)

    # Final safety: never return an invalid / non-finite path.
    if not (best.shape[1] == 3 and best.shape[0] >= 1 and np.all(np.isfinite(best))):
        return seed
    if _compute_value(best) == 0:
        return seed
    return best


# EVOLVE-BLOCK-END
