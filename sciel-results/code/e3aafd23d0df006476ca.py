import time
import math
import numpy as np

_T0 = time.time()
_DEADLINE = 620.0  # safety margin before 630s kill


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _laguerre_row(k, x):
    """L_k(x) for k in [0, maxk], computed via stable 3-term recurrence."""
    maxk = int(k)
    x = np.asarray(x, dtype=float)
    out = np.empty((maxk + 1,) + x.shape, dtype=float)
    L0 = np.ones_like(x)
    L1 = 1.0 - x
    out[0] = L0
    if maxk >= 1:
        out[1] = L1
    for n in range(1, maxk):
        L2 = ((2 * n + 1) * L1 - n * L0 - (n + 1) * x * L1) / (n + 1)
        out[n + 1] = L2
        L0, L1 = L1, L2
    return out


def _eval_P(r, maxk):
    """Return P(r) and P'(r) at all roots r (12 double + 1 single)."""
    R = len(r)
    # coefficients c_k = sum_{j>=k} w_j / k!,  w_j = 1/j!
    w = np.array([1.0 / math.factorial(j) for j in range(R)])
    c = np.empty(maxk + 1)
    for k in range(maxk + 1):
        c[k] = w[k:].sum() / math.factorial(k)
    rows = _laguerre_row(maxk, r)   # shape (maxk+1, R)
    P = rows.T @ c                  # (R,)
    drows = _laguerre_row(maxk - 1, r)  # shape (maxk, R)
    Pp = drows.T @ c                # (R,)
    return P, Pp


def _nonneg_penalty(r, maxk, npts=6000):
    """Penalize negative values of P on [0, r_max]."""
    rmax = max(r[-1], 1.0) * 1.02
    grid = np.linspace(0.0, rmax, npts)
    w = np.array([1.0 / math.factorial(j) for j in range(len(r))])
    c = np.empty(maxk + 1)
    for k in range(maxk + 1):
        c[k] = w[k:].sum() / math.factorial(k)
    rows = _laguerre_row(maxk, grid)
    vals = rows.T @ c
    neg = np.clip(-vals, 0.0, None)
    return float((neg * neg).sum())


def _solve(maxk, r0, iters=8000, damp=0.35, tol=1e-9):
    r = np.array(r0, dtype=float)
    R = len(r)
    best_r = r.copy()
    best_obj = np.inf
    for it in range(iters):
        if _time_left() < 12.0:
            break
        P, Pp = _eval_P(r, maxk)
        resid = np.concatenate([P[:R - 1], Pp[:R - 1], P[R - 1]])
        obj = float(resid @ resid) + 1e-3 * _nonneg_penalty(r, maxk, 1500)
        if obj < best_obj:
            best_obj = obj
            best_r = r.copy()
        if obj < tol:
            break
        # build Jacobian numerically
        J = np.zeros((2 * R - 1, R))
        h = 1e-6
        for j in range(R):
            rp = r.copy()
            rp[j] += h
            Pp2, Ppp2 = _eval_P(rp, maxk)
            rm = r.copy()
            rm[j] -= h
            Pm2, Ppm2 = _eval_P(rm, maxk)
            dP = (Pp2 - Pm2) / (2 * h)
            dPp = (Ppp2 - Ppm2) / (2 * h)
            J[:R - 1, j] = dP[:R - 1]
            J[R - 1:2 * R - 2, j] = dPp[:R - 1]
            J[2 * R - 2, j] = dP[R - 1]
        try:
            step, *_ = np.linalg.lstsq(J, -resid, rcond=None)
        except Exception:
            step = np.zeros(R)
        step = np.clip(step, -0.5, 0.5)
        r_new = r + damp * step
        r_new = np.clip(r_new, 1e-6, None)
        r_new = np.maximum.accumulate(r_new)  # keep increasing
        r = r_new
    return best_r


def construct():
    m = 12
    maxk = 2 * m + 1  # 25
    R = maxk + 1      # 25 roots: 12 double + 1 single
    # initial guess: refined seed spacing
    r0 = 4.0 * np.arange(1, R + 1, dtype=float)
    r0 = np.maximum.accumulate(r0)

    best = r0.copy()
    best_score = float(best[-1])

    # primary solve
    try:
        r = _solve(maxk, r0, iters=6000)
        s = float(r[-1])
        if s < best_score:
            best, best_score = r.copy(), s
    except Exception:
        pass

    if _time_left() < 30.0:
        return [float(v) for v in best]

    # second restart from a compressed initial guess
    try:
        r0b = 2.0 * np.arange(1, R + 1, dtype=float)
        r0b = np.maximum.accumulate(r0b)
        r = _solve(maxk, r0b, iters=4000)
        s = float(r[-1])
        if s < best_score:
            best, best_score = r.copy(), s
    except Exception:
        pass

    return [float(v) for v in best]
