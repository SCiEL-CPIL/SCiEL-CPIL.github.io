"""Rewritten program: minimize max|autoconv(f)| / (sum f)^2 for signed step heights.

Strategy (structural change from the constant seed):
  * The objective is scale-invariant, so we normalize each candidate to unit integral.
  * The raw max-abs-autoconvolution is non-smooth; we optimize a stable soft-max
    surrogate (log-sum-exp of |A[k]|) with L-BFGS-B, which has a good gradient.
  * Diverse starts: constant, many chirps (cos tones + DC), smoothed random, and
    two-sign block functions. Each is polished; the best by the *true* metric wins.
  * Explicit wall-clock deadline guard returns the best result found with margin.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
DEADLINE = 140.0          # wall-clock safety margin (evaluator kills at 150 s)
_T0 = time.time()
_T = 80.0                 # soft-max temperature (large => close to true max)

def _remaining():
    return DEADLINE - (time.time() - _T0)

def _autoconv(f):
    return np.convolve(f, f, mode="full")

def _true_value(f):
    s = float(np.sum(f))
    if abs(s) < 1e-9:
        return 1e12
    return float(np.max(np.abs(_autoconv(f / s))))

def _surrogate(f):
    """Stable soft-max of |A[k]| at unit-integral scale; smooth for L-BFGS-B."""
    s = float(np.sum(f))
    if abs(s) < 1e-9:
        return 1e12
    a = np.abs(_autoconv(f / s))
    m = float(a.max())
    # log-sum-exp trick: m + (1/T) log sum exp(T (a - m))
    e = np.exp(_T * (a - m))
    return m + (1.0 / _T) * float(np.log(e.sum()))

_BOUNDS = [(-20.0, 20.0)] * N

def _polish(f0):
    if _remaining() < 6.0:
        return f0
    try:
        res = minimize(_surrogate, f0, method="L-BFGS-B", bounds=_BOUNDS,
                       options={"maxiter": 150, "maxfun": 2500,
                                "ftol": 1e-14, "gtol": 1e-9})
        return res.x
    except Exception:
        return f0

def construct():
    rng = np.random.default_rng(12345)
    i = np.arange(N)

    starts = []
    # constant (the seed shape), unit integral
    starts.append(np.ones(N) / N)
    # chirps: tone + DC so the integral is not near zero, then unit-integral
    for f0 in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
        f = np.cos(2.0 * np.pi * f0 * i / N) + 0.6
        s = float(f.sum())
        if abs(s) > 1e-6:
            starts.append(f / s)
    # smoothed random fields (both signs)
    for _ in range(6):
        f = np.convolve(rng.standard_normal(N), np.ones(25) / 25.0, mode="same")
        s = float(f.sum())
        if abs(s) > 1e-6:
            starts.append(f / s)
    # two-sign block functions
    for split, amp in [(0.3, -0.5), (0.5, -0.5), (0.7, -0.5), (0.5, -1.0)]:
        f = np.ones(N)
        f[int(N * split):] = amp
        s = float(f.sum())
        if abs(s) > 1e-6:
            starts.append(f / s)

    starts = np.array(starts)

    best_f = starts[0].copy()
    best_val = _true_value(best_f)

    # rank starts by true value, polish the promising ones first
    order = np.argsort([_true_value(s) for s in starts])
    for idx in order:
        if _remaining() < 10.0:
            break
        f1 = _polish(starts[idx])
        v1 = _true_value(f1)
        if v1 < best_val:
            best_val = v1
            best_f = f1
        # theoretical floor for the unit-integral max-abs-autoconv is 1/N
        if best_val <= (1.0 / N) * (1.0 + 1e-9):
            break

    # return signed step heights (scale is irrelevant for the ratio)
    s = float(np.sum(best_f))
    if abs(s) > 1e-9:
        best_f = best_f / s
    return best_f
