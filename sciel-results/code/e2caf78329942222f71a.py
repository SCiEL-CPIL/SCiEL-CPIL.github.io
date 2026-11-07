"""Uncertainty inequality: minimize the largest sign-changing root of q = g/x^2
subject to g(x) >= 0 for all x >= 0. Uses a smooth log-sum-exp surrogate for
the root, a dense-grid positivity constraint, and an exact final polish."""

# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = time.time() + 190.0  # safety margin vs. 210 s wall-clock kill
_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

# Hermite values at 0 (physicists' convention): H_{2k}(0) = (-1)^k (2k)! / k!
_H0_0 = 1.0
_H4_0 = -12.0
_H8_0 = 1680.0
_H12_0 = -4790016.0

# grid of x^2 values (x = sqrt)
_X2 = np.linspace(0.0, 220.0, 1500)
_SQRT_X2 = np.sqrt(_X2)


def _coeffs_to_g(c):
    c0, c1, c2 = c
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    return c0, c1, c2, c3


def _g_values(c, x2):
    c0, c1, c2, c3 = _coeffs_to_g(c)
    x2 = np.asarray(x2, dtype=float)
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    x12 = x6 * x6
    return (c0 * x12 + c1 * x8 + c2 * x6 + c3 * x4) / x2


def _q_values(c, x2):
    c0, c1, c2, c3 = _coeffs_to_g(c)
    x2 = np.asarray(x2, dtype=float)
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    return c0 * x6 + c1 * x4 + c2 * x2 + c3


def _exact_value(c):
    """Exact value = r^2/(2*pi), r = largest real root of q where q changes sign.
    Returns None if no sign-changing root (g >= 0 everywhere => value 0)."""
    c0, c1, c2, c3 = _coeffs_to_g(c)
    # q = c0 x^6 + c1 x^4 + c2 x^2 + c3, let y = x^2:
    # p(y) = c0 y^3 + c1 y^2 + c2 y + c3
    if abs(c0) > 1e-300:
        roots = np.roots([c0, c1, c2, c3])
    elif abs(c1) > 1e-300:
        roots = np.roots([c1, c2, c3])
    elif abs(c2) > 1e-300:
        roots = np.array([-c3 / c2])
    else:
        return None
    best = None
    eps = 1e-7
    for y in roots:
        if abs(y.imag) > 1e-6 * max(1.0, abs(y.real)):
            continue
        y = float(y.real)
        if y < 0.0:
            continue
        # sign change of p around y
        lo = max(0.0, y - eps)
        hi = y + eps
        plo = c0 * lo**3 + c1 * lo**2 + c2 * lo + c3
        phi = c0 * hi**3 + c1 * hi**2 + c2 * hi + c3
        if plo * phi < 0.0:
            if best is None or y > best:
                best = y
    if best is None:
        return None
    return best / (2.0 * math.pi)


def _surrogate(c, beta):
    """Smooth max over grid of log(q) (in y-space) with positivity penalty."""
    q = _q_values(c, _X2)
    qpos = np.clip(q, 1e-30, None)
    m = np.max(np.log(qpos))
    lse = m + np.log(np.sum(np.exp(np.log(qpos) - m)))
    # positivity penalty: sum of log(1 + exp(-beta * q)) ~ sum of soft negative parts
    pen = np.sum(np.log1p(np.exp(-beta * q)))
    return lse + beta * pen


def _feasible(c):
    q = _q_values(c, _X2)
    return float(np.min(q)) >= -1e-9


def construct():
    t_start = time.time()
    best_c = _SEED.copy()
    best_val = _exact_value(_SEED)
    if best_val is None:
        best_val = 0.0

    def _maybe_update(c):
        nonlocal best_c, best_val
        if not _feasible(c):
            return
        v = _exact_value(c)
        if v is None:
            v = 0.0
        if v < best_val - 1e-12:
            best_val = v
            best_c = c.copy()

    def _deadline():
        return time.time() > _DEADLINE - 20.0

    # ---- Phase 1: Nelder-Mead on smooth surrogate ----
    try:
        from scipy.optimize import minimize
    except Exception:
        minimize = None

    if minimize is not None and not _deadline():
        x0 = _SEED.copy()
        for beta in (8.0, 20.0, 60.0):
            if _deadline():
                break
            try:
                res = minimize(
                    lambda c: _surrogate(c, beta),
                    x0,
                    method="Nelder-Mead",
                    options={"maxiter": 3000, "xatol": 1e-10, "fatol": 1e-12},
                )
                c = np.asarray(res.x, dtype=float)
                if np.all(np.isfinite(c)):
                    _maybe_update(c)
                    x0 = c.copy()
            except Exception:
                pass

    # ---- Phase 2: local polish with L-BFGS-B (feasible refinement) ----
    if minimize is not None and not _deadline():
        # objective = surrogate root only (no penalty); use current best as start
        try:
            res = minimize(
                lambda c: _surrogate(c, 60.0),
                best_c,
                method="L-BFGS-B",
                options={"maxiter": 2000, "gtol": 1e-10},
            )
            c = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(c)) and _feasible(c):
                _maybe_update(c)
        except Exception:
            pass

    # ---- Phase 3: exact final polish on the largest root via root tracking ----
    if not _deadline():
        for it in range(200):
            if _deadline():
                break
            c = best_c.copy()
            # try a small local Newton-ish step: perturb each coord, keep best
            improved = False
            for scale in (1e-3, 1e-4, 1e-5):
                for k in range(3):
                    for sgn in (1.0, -1.0):
                        c2 = c.copy()
                        c2[k] += sgn * scale * (abs(c[k]) + 1e-3)
                        if _feasible(c2):
                            v = _exact_value(c2)
                            if v is None:
                                v = 0.0
                            if v < best_val - 1e-12:
                                best_val = v
                                best_c = c2.copy()
                                improved = True
            if not improved:
                break

    # ---- Final exact evaluation of the best candidate ----
    v = _exact_value(best_c)
    if v is None:
        v = 0.0
    return np.array([best_c[0], best_c[1], best_c[2]], dtype=float)
# EVOLVE-BLOCK-END
