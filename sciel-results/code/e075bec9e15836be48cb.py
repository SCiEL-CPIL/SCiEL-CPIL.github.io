"""Pack 300 equal circles in the unit disk; maximize common radius.

Design (hex-lattice family, refined):
  * build a hexagonal lattice (with an optional rotated "offset" variant)
    and optionally add a small random jitter to break lattice degeneracy;
  * local-search the smooth softmin surrogate with L-BFGS-B from several
    starts, then polish the true nonsmooth objective with COBYLA;
  * deterministic (fixed seeds) with an explicit wall-clock deadline guard
    that returns the best valid configuration found so far.
"""
import time
import numpy as np
from scipy.optimize import minimize, cdist
from scipy.spatial import cKDTree

N = 300

# ---- time budget -----------------------------------------------------------
_T0 = time.time()
_DEADLINE = _T0 + 112.0          # safety margin vs 120 s evaluator
_EPS = 1e-12


def _now():
    return time.time()


def _radius(p):
    """Common radius of a configuration (True objective)."""
    if p.shape[0] < 2:
        return float(np.inf)
    d = cdist(p, p)
    iu = np.triu_indices_from(d, k=1)
    min_pair = d[iu].min()
    return 0.5 * min(min_pair, 1.0 - np.linalg.norm(p, axis=1).max())


def _objective(p):
    """Smooth softmin surrogate (larger is better)."""
    if p.shape[0] < 2:
        return 0.0
    d = cdist(p, p)
    iu = np.triu_indices_from(d, k=1)
    m = d[iu].min()
    m2 = d[iu].min(1)
    mn = 1.0 - np.linalg.norm(p, axis=1).max()
    v = np.array([m, m2, mn])
    v = np.maximum(v, 1e-9)
    return 0.5 * (v.min() - np.log(v.sum()) + np.log(3.0))


def _softmin_val(p, k):
    """Softmin of the three limiting quantities (used for COBYLA)."""
    if p.shape[0] < 2:
        return 0.0
    d = cdist(p, p)
    iu = np.triu_indices_from(d, k=1)
    m = d[iu].min()
    m2 = d[iu].min(1)
    mn = 1.0 - np.linalg.norm(p, axis=1).max()
    v = np.array([m, m2, mn])
    v = np.maximum(v, 1e-9)
    return 0.5 * (v.min() - (1.0 / k) * np.log(np.exp(-k * v).sum()) + (1.0 / k) * np.log(3.0))


def _make_hex(spacing, jitter, rng):
    """Hexagonal lattice points inside the disk, optionally jittered."""
    k = int(1.0 / spacing) + 3
    xs, ys = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))
    j = ys
    pts = np.stack(((xs + 0.5 * (j % 2)) * spacing,
                    j * spacing * np.sqrt(3.0) / 2.0), axis=-1).reshape(-1, 2)
    r = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts[r <= 1.0]
    if jitter > 0:
        pts = pts + jitter * rng.standard_normal(pts.shape)
    return pts


def _initial(seed):
    """Generate one starting configuration of exactly N points."""
    rng = np.random.default_rng(seed)
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 3
        xs, ys = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))
        j = ys
        if seed % 2 == 0:
            pts = np.stack(((xs + 0.5 * (j % 2)) * spacing,
                            j * spacing * np.sqrt(3.0) / 2.0), axis=-1).reshape(-1, 2)
        else:
            # rotated / offset lattice: shift every other column by half a row
            pts = np.stack((xs * spacing,
                            (j + 0.5 * (xs % 2)) * spacing * np.sqrt(3.0) / 2.0),
                           axis=-1).reshape(-1, 2)
        r = np.hypot(pts[:, 0], pts[:, 1])
        pts = pts[r <= 1.0]
        if len(pts) >= N:
            jitter = 0.04 * spacing * rng.random()
            if jitter > 0:
                pts = pts + jitter * rng.standard_normal(pts.shape)
            order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
            return pts[order[:N]].copy()
        spacing *= 0.99


def _lbfgs(p0, k, iters):
    res = minimize(_objective, p0, method="L-BFGS-B",
                   jac=True, options={"maxiter": iters,
                                      "ftol": 1e-12, "gtol": 1e-9})
    return res.x


def _polish(p, iters):
    def f(x):
        return -_softmin_val(x, 250.0)
    res = minimize(f, p, method="COBYLA", options={"maxiter": iters,
                                                   "rhobeg": 0.02, "tol": 1e-12})
    return res.x


def _valid(p):
    if p.shape != (N, 2) or not np.all(np.isfinite(p)):
        return False
    if np.linalg.norm(p, axis=1).max() > 1.0 + 1e-9:
        return False
    d = cdist(p, p)
    iu = np.triu_indices_from(d, k=1)
    return d[iu].min() > 1e-9


def construct():
    t0 = _now()

    def time_ok():
        return (_now() - t0) < (_DEADLINE - t0)

    # ---- multi-start local search ------------------------------------------
    best = None
    best_r = -1.0
    n_starts = 10
    iters = 200
    for s in range(n_starts):
        if not time_ok():
            break
        p0 = _initial(s)
        if not _valid(p0):
            continue
        p = _lbfgs(p0, 200.0, iters)
        if not _valid(p):
            continue
        r = _radius(p)
        if r > best_r:
            best_r, best = r, p.copy()

    # ---- adaptive restarts from the best so far ----------------------------
    restarts = 0
    while best is not None and restarts < 40 and time_ok():
        rng = np.random.default_rng(1000 + restarts)
        sigma = 0.015 * max(best_r, 1e-3)
        p0 = best + sigma * rng.standard_normal(best.shape)
        p = _lbfgs(p0, 250.0, 120)
        if not _valid(p):
            continue
        r = _radius(p)
        if r > best_r + 1e-12:
            best_r, best = r, p.copy()
            restarts += 1
        else:
            restarts += 1

    # ---- final high-precision polish ---------------------------------------
    if best is not None and time_ok():
        p = _polish(best, 300)
        if _valid(p) and _radius(p) > best_r:
            best, best_r = p.copy(), _radius(p)

    # ---- safety net: return a valid config even if search failed ----------
    if best is None:
        # fallback to a guaranteed-valid hex lattice
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 3
            xs, ys = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))
            j = ys
            pts = np.stack(((xs + 0.5 * (j % 2)) * spacing,
                            j * spacing * np.sqrt(3.0) / 2.0), axis=-1).reshape(-1, 2)
            r = np.hypot(pts[:, 0], pts[:, 1])
            pts = pts[r <= 1.0]
            if len(pts) >= N:
                order = np.argsort(r, kind="stable")
                return pts[order[:N]]
            spacing *= 0.99

    return np.asarray(best, dtype=float)
