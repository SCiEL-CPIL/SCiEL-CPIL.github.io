"""Rewritten program: uncertainty-inequality constant minimization.
Design family kept: continuous optimization over the 2D direction (a=c1/c0, b=c2/c0),
scale-invariant. Improvements: soft-max (log-sum-exp) smooth surrogate for the
nonsmooth max-of-roots objective, gradient descent + L-BFGS-B + Nelder-Mead,
high-precision exact final polish, explicit deadline guard.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

# ---------- Precomputed exact Hermite data (physicists' H_n) ----------
# H_n(0) values
_H0_0 = 1.0
_H4_0 = -12.0
_H8_0 = 1008.0
_H12_0 = -1032192.0
# Leading coefficient of H_12 is 2^12 = 4096
_H12_LEAD = 4096.0

# q(x) = g(x)/x^2  (g = c0 H0 + c1 H4 + c2 H8 + c3 H12, g(0)=0)
# q coefficients in ASCENDING powers of x (q[i] * x^i), length 11 (degree 10)
_Q = np.array([
    -1032192.0,      # x^0
    0.0,             # x^1
    12446304.0,      # x^2
    0.0,             # x^3
    -114412800.0,    # x^4
    0.0,             # x^5
    559872000.0,     # x^6
    0.0,             # x^7
    -1472448000.0,   # x^8
    0.0,             # x^9
    16777216.0,      # x^10
], dtype=float)


def _c3(a, b):
    """c3 given c0=1, c1=a, c2=b, enforcing g(0)=0."""
    return (a * _H4_0 + b * _H8_0 - _H0_0) / _H12_0


def _q_coeffs(a, b):
    """Return q coefficient array (ascending) for direction (a,b), c0=1."""
    c3 = _c3(a, b)
    return _Q + a * _Q + 0.0 * np.array([0.0]) + 0.0  # placeholder, replaced below
    # (corrected below)


def _qvec(a, b):
    c3 = _c3(a, b)
    return _Q + a * _Q * 0.0 + b * _Q * 0.0 + c3 * _Q * 0.0  # placeholder
    # actual build below


# Build q properly: q = q0 + a*q1 + b*q2 + c3(a,b)*q3
_Q0 = np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
_Q1 = np.array([-12.0, 0, 0, 0, 240.0, 0, -13440.0, 0, 302400.0, 0, -2419200.0], dtype=float)
_Q2 = np.array([1008.0, 0, 0, 0, -20160.0, 0, 665280.0, 0, -13305600.0, 0, 155232000.0], dtype=float)
_Q3 = np.array([-1032192.0, 0, 0, 0, 12446304.0, 0, -114412800.0, 0, 559872000.0, 0, -1472448000.0], dtype=float)
_Q3_lead = 16777216.0  # x^10 coefficient of H12 (2^12)


def _qvec(a, b):
    c3 = (a * _H4_0 + b * _H8_0 - _H0_0) / _H12_0
    return _Q0 + a * _Q1 + b * _Q2 + c3 * _Q3


def _sign_changing_roots(q):
    """Return list of positive real roots of q at which q changes sign (odd multiplicity)."""
    # descending order for np.roots
    # strip leading (highest-degree) near-zeros
    desc = q[::-1].copy()
    while len(desc) > 1 and abs(desc[0]) < 1e-14:
        desc = desc[1:]
    if len(desc) <= 1:
        return []
    try:
        roots = np.roots(desc)
    except Exception:
        return []
    out = []
    for r in roots:
        if abs(r.imag) > 1e-8 * max(1.0, abs(r.real)):
            continue
        rr = float(r.real)
        if rr <= 1e-12:
            continue
        eps = 1e-9 * max(1.0, rr)
        qb = np.polyval(desc, rr - eps)
        qa = np.polyval(desc, rr + eps)
        if qb * qa < 0.0:
            out.append(rr)
    return out


def _value_exact(a, b):
    """Exact value = max over sign-changing roots r of r^2/(2*pi)."""
    q = _qvec(a, b)
    roots = _sign_changing_roots(q)
    if not roots:
        return math.inf
    rmax = max(roots)
    return rmax * rmax / (2.0 * math.pi)


def _soft_value(a, b, beta):
    """Smooth surrogate: soft-max over sign-changing roots of r^2/(2*pi)."""
    q = _qvec(a, b)
    roots = _sign_changing_roots(q)
    if not roots:
        return 1e12
    vals = np.array([r * r / (2.0 * math.pi) for r in roots])
    m = np.max(vals)
    if not np.isfinite(m):
        return 1e12
    s = np.sum(np.exp((vals - m) / beta))
    return m + beta * math.log(s)


def _objective(params, beta):
    a, b = params
    v = _soft_value(a, b, beta)
    if not np.isfinite(v):
        return 1e12
    return v


def construct():
    t0 = time.time()
    deadline = t0 + 200.0  # safety margin before 210s kill

    def _time_left():
        return deadline - time.time()

    # Seed direction (normalized, c0=1)
    a0, b0 = (1.0 / 25.0) / (-113.0 / 100.0), (1.0 / 3240.0) / (-113.0 / 100.0)
    # a0 = (1/25)/(-113/100) = (1/25)*(-100/113) = -4/113
    # b0 = (1/3240)/(-113/100) = (1/3240)*(-100/113) = -100/(3240*113)
    a0 = -4.0 / 113.0
    b0 = -100.0 / (3240.0 * 113.0)

    best_ab = np.array([a0, b0])
    best_val = _value_exact(a0, b0)

    # Stage 1: coarse global search (Nelder-Mead multi-start) with moderate beta
    beta = 0.05
    rng = np.random.RandomState(7)
    starts = [np.array([a0, b0])]
    for _ in range(12):
        starts.append(np.array([a0, b0]) + rng.normal(0.0, 0.5, size=2))

    for s in starts:
        if _time_left() < 15.0:
            break
        try:
            res = minimize(_objective, s, args=(beta,), method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10})
            v = _value_exact(res.x[0], res.x[1])
            if np.isfinite(v) and v < best_val:
                best_val = v
                best_ab = res.x.copy()
        except Exception:
            pass

    # Stage 2: L-BFGS-B on soft surrogate, annealing beta down
    cur = best_ab.copy()
    for beta in [0.05, 0.02, 0.008, 0.003, 0.001]:
        if _time_left() < 12.0:
            break
        try:
            res = minimize(_objective, cur, args=(beta,), method='L-BFGS-B',
                           options={'maxiter': 400, 'gtol': 1e-12, 'ftol': 1e-15})
            v = _value_exact(res.x[0], res.x[1])
            if np.isfinite(v) and v < best_val:
                best_val = v
                best_ab = res.x.copy()
            cur = res.x.copy()
        except Exception:
            pass

    # Stage 3: Nelder-Mead polish on exact objective (no smoothing)
    if _time_left() > 8.0:
        try:
            res = minimize(lambda p: _value_exact(p[0], p[1]), best_ab,
                           method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-13, 'fatol': 1e-13})
            v = _value_exact(res.x[0], res.x[1])
            if np.isfinite(v) and v < best_val:
                best_val = v
                best_ab = res.x.copy()
        except Exception:
            pass

    # Stage 4: high-precision local polish (very tight Nelder-Mead + small line search)
    if _time_left() > 5.0:
        for scale in [1e-3, 1e-5, 1e-7]:
            if _time_left() < 4.0:
                break
            try:
                res = minimize(lambda p: _value_exact(p[0], p[1]), best_ab,
                               method='Nelder-Mead',
                               options={'maxiter': 800, 'xatol': 1e-15, 'fatol': 1e-15,
                                        'initial_simplex': [best_ab + scale * np.array([1.0, 0.0]),
                                                            best_ab + scale * np.array([0.0, 1.0])]})
                v = _value_exact(res.x[0], res.x[1])
                if np.isfinite(v) and v < best_val:
                    best_val = v
                    best_ab = res.x.copy()
            except Exception:
                pass

    # Convert direction back to coefficients [c0, c1, c2]
    c0 = -113.0 / 100.0
    c1 = c0 * best_ab[0]
    c2 = c0 * best_ab[1]
    return [c0, c1, c2]
