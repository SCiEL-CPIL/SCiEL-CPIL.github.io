# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# ---------- Sofa grid ----------
_U = np.arange(-3.0, 1.5, 0.01)
_V = np.arange(-1.5, 1.5, 0.01)
_SX, _SY = np.meshgrid(_U, _V, indexing='ij')
_SX = _SX.ravel()
_SY = _SY.ravel()
_NSOFA = _SX.size

# ---------- Deadline ----------
_T0 = None
_LIMIT = 600.0
_MARGIN = 30.0


def _elapsed():
    return time.time() - _T0 if _T0 is not None else 0.0


def _over():
    return _elapsed() > _LIMIT - _MARGIN


# ---------- Geometry ----------
def _placements(path):
    """path: (n,3) user waypoints. Prepend (0,0,0), append (-2,0,90).
    Densify: between consecutive, m = max(1, ceil(|da|)) steps (so m+1 intervals).
    Returns (M,3) array."""
    P = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    for r in path:
        P = np.vstack([P, r])
    P = np.vstack([P, [-2.0, 0.0, 90.0]])
    # drop exact duplicates in angle to avoid m=0 issues; keep
    out = [P[0]]
    for i in range(1, len(P)):
        a1, a2 = P[i - 1, 2], P[i, 2]
        m = max(1, int(np.ceil(abs(a2 - a1) - 1e-9)))
        for k in range(1, m + 1):
            t = k / m
            out.append([
                P[i - 1, 0] + t * (P[i, 0] - P[i - 1, 0]),
                P[i - 1, 1] + t * (P[i, 1] - P[i - 1, 1]),
                a1 + t * (a2 - a1),
            ])
    return np.array(out, dtype=np.float64)


def _in_base(u, v):
    """base L: (0<=u<=1 & -3<=v<=1) OR (-3<=u<=1 & 0<=v<=1). vectorized."""
    return ((u >= -1e-9) & (u <= 1 + 1e-9) & (v >= -3 - 1e-9) & (v <= 1 + 1e-9)) | \
           ((u >= -3 - 1e-9) & (u <= 1 + 1e-9) & (v >= -1e-9) & (v <= 1 + 1e-9))


def _sofa_mask(placements):
    """Return boolean mask of sofa points (in every placement)."""
    keep = np.ones(_NSOFA, dtype=bool)
    for (x, y, a) in placements:
        rad = np.deg2rad(a)
        c, s = np.cos(rad), np.sin(rad)
        # w - (x,y)
        wx = _SX - x
        wy = _SY - y
        # R(-a): rotate by -a
        u = c * wx + s * wy
        v = -s * wx + c * wy
        ok = _in_base(u, v)
        keep &= ok
        if not keep.any():
            break
    return keep


def _area(placements):
    return _sofa_mask(placements).sum() * 1e-4


def _softmin_area(placements, lam=30.0):
    """Smooth surrogate: for each point, min over placements of signed-distance-like
    indicator. Use: point in all iff min over placements of 'inside' is True.
    Smooth: for each placement compute a smooth inside score in [0,1], take soft-min
    via -logsumexp(-s_i)/lam. Sum over points."""
    scores = []
    for (x, y, a) in placements:
        rad = np.deg2rad(a)
        c, s = np.cos(rad), np.sin(rad)
        wx = _SX - x
        wy = _SY - y
        u = c * wx + s * wy
        v = -s * wx + c * wy
        # inside base: union of two rects. smooth inside via rect distance.
        # rect1: u in [0,1], v in [-3,1]; rect2: u in [-3,1], v in [0,1]
        d1 = np.maximum(np.maximum(0.0 - u, u - 1.0), np.maximum(-3.0 - v, v - 1.0))
        d2 = np.maximum(np.maximum(-3.0 - u, u - 1.0), np.maximum(0.0 - v, v - 1.0))
        d = np.minimum(d1, d2)  # distance to base (<=0 inside)
        sc = 1.0 / (1.0 + np.exp(lam * d))  # ~1 inside, ~0 outside
        scores.append(sc)
    S = np.stack(scores, axis=0)  # (M, N)
    # soft-min across placements
    sm = -np.logsumexp(-lam * S, axis=0) / lam
    return sm.sum()


# ---------- Objective ----------
def _objective(params):
    n = len(params) // 3
    path = params.reshape(n, 3)
    pl = _placements(path)
    if _over():
        return 0.0
    return -_softmin_area(pl)


def _true_objective(params):
    n = len(params) // 3
    path = params.reshape(n, 3)
    pl = _placements(path)
    return -_area(pl)


# ---------- Baseline path ----------
def _baseline_params(n):
    # Start from a known-good arc, resampled to n control points.
    base = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ])
    # resample angles uniformly in [15,86] to n points, and interpolate x,y
    ang = np.linspace(base[0, 2], base[-1, 2], n)
    xs = np.interp(ang, base[:, 2], base[:, 0])
    ys = np.interp(ang, base[:, 2], base[:, 1])
    return np.stack([xs, ys, ang], axis=1).ravel()


# ---------- Main ----------
def moving_sofa_2d() -> np.ndarray:
    global _T0
    _T0 = time.time()

    best_path = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)
    best_area = _area(_placements(best_path))

    n = 12
    p0 = _baseline_params(n)

    # Phase 1: Nelder-Mead on smooth surrogate
    try:
        res = minimize(
            _objective, p0, method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-5, 'adaptive': True},
        )
        if not _over() and res.fun < 0:
            cand = res.x.reshape(n, 3)
            a = _area(_placements(cand))
            if a > best_area:
                best_area = a
                best_path = cand
    except Exception:
        pass

    # Phase 2: polish with true objective, tighter
    if not _over():
        try:
            res2 = minimize(
                _true_objective, best_path.ravel(), method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True},
            )
            cand = res2.x.reshape(n, 3)
            a = _area(_placements(cand))
            if a > best_area:
                best_area = a
                best_path = cand
        except Exception:
            pass

    # Phase 3: local coordinate perturbation (coordinate ascent) on true objective
    if not _over():
        cur = best_path.copy()
        cur_area = best_area
        step_xy = 0.05
        step_a = 2.0
        improved = True
        while improved and not _over():
            improved = False
            for i in range(len(cur)):
                for d in (1, -1):
                    for (dx, dy, da) in [(step_xy, 0, 0), (0, step_xy, 0), (0, 0, step_a)]:
                        trial = cur.copy()
                        trial[i, 0] += d * dx
                        trial[i, 1] += d * dy
                        trial[i, 2] += d * da
                        a = _area(_placements(trial))
                        if a > cur_area + 1e-9:
                            cur = trial
                            cur_area = a
                            improved = True
            # shrink steps a bit
            step_xy *= 0.7
            step_a *= 0.7
            if step_xy < 1e-4:
                break
        if cur_area > best_area:
            best_area = cur_area
            best_path = cur

    # Ensure angles within a sensible range and finite
    best_path = np.nan_to_num(best_path, nan=0.0, posinf=90.0, neginf=0.0)
    if best_path.shape[0] < 1:
        best_path = np.array([[-0.4, 0.3, 45.0]])
    return best_path


# EVOLVE-BLOCK-END
