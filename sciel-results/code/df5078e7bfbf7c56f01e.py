"""Search for coefficients minimizing r^2/(2*pi) where r is the largest
sign-changing real root of q = g/x^2, subject to g >= 0."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import hermite


def _hermite_coeffs(n):
    """Return coefficients of physicists' Hermite polynomial H_n(x) in ascending powers."""
    coeffs = np.zeros(n + 1)
    if n == 0:
        coeffs[0] = 1.0
        return coeffs
    if n == 1:
        coeffs[1] = 2.0
        return coeffs
    h_prev = np.array([1.0])
    h_curr = np.array([0.0, 2.0])
    for k in range(2, n + 1):
        h_next = np.zeros(k + 1)
        h_next[1:] += 2.0 * h_curr
        h_next[:len(h_prev)] -= 2.0 * k * h_prev
        h_prev, h_curr = h_curr, h_next
    return h_curr


# Precompute Hermite coefficients
_H0 = _hermite_coeffs(0)
_H4 = _hermite_coeffs(4)
_H8 = _hermite_coeffs(8)
_H12 = _hermite_coeffs(12)
_H0_0 = _H0[0]
_H4_0 = _H4[0]
_H8_0 = _H8[0]
_H12_0 = _H12[0]
_H12_lead = _H12[-1]  # 2^12 = 4096


def _build_q(c0, c1, c2):
    """Return coefficients of q(x)=g(x)/x^2 (ascending powers) as a float array, and c3."""
    c0 = float(c0); c1 = float(c1); c2 = float(c2)
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    
    g = np.zeros(13)
    g[:len(_H0)] += c0 * _H0
    g[:len(_H4)] += c1 * _H4
    g[:len(_H8)] += c2 * _H8
    g[:len(_H12)] += c3 * _H12
    
    q = np.zeros(11)
    q[:] = g[2:]
    return q, c3


def _signchange_root(q):
    """Largest real root of q where q changes sign."""
    nz = np.nonzero(q)[0]
    if len(nz) == 0:
        return None
    q_trimmed = q[nz[0]:]
    roots = np.roots(q_trimmed)
    real = [z.real for z in roots if abs(z.imag) < 1e-9 * max(1.0, abs(z))]
    if not real:
        return None
    best = None
    for r in real:
        eps = 1e-7
        q_left = np.polyval(q, r - eps)
        q_right = np.polyval(q, r + eps)
        if q_left * q_right < 0:
            if best is None or r > best:
                best = r
    return best


def _eval_g(c0, c1, c2, xs, c3=None):
    """Evaluate g(x) at points xs."""
    if c3 is None:
        _, c3 = _build_q(c0, c1, c2)
    g = np.zeros(13)
    g[:len(_H0)] += c0 * _H0
    g[:len(_H4)] += c1 * _H4
    g[:len(_H8)] += c2 * _H8
    g[:len(_H12)] += c3 * _H12
    return np.polyval(g, xs)


def _objective(c):
    """Smooth objective with correct Hermite values."""
    q, c3 = _build_q(c[0], c[1], c[2])
    g_lead = c3 * _H12_lead
    if g_lead < 0:
        g_lead_eff = -g_lead
    else:
        g_lead_eff = g_lead

    r = _signchange_root(q)
    if r is None:
        r = 0.0
    obj = r * r

    xs = np.linspace(0.0, 20.0, 400)
    g = _eval_g(c[0], c[1], c[2], xs, c3)
    if g_lead < 0:
        g = -g
    mn = float(np.min(g))
    if mn < 0:
        obj += 50.0 * (mn / max(1.0, abs(r))) ** 2
    if g_lead < 0:
        obj += 10.0 * g_lead ** 2
    return obj


def _value(c):
    """Exact value = r^2/(2*pi) for the sign-changing root; None if invalid."""
    q, c3 = _build_q(c[0], c[1], c[2])
    g_lead = c3 * _H12_lead
    r = _signchange_root(q)
    if r is None:
        return None
    xs = np.linspace(0.0, 20.0, 600)
    g = _eval_g(c[0], c[1], c[2], xs, c3)
    if g_lead < 0:
        g = -g
    if float(np.min(g)) < -1e-9:
        return None
    return r * r / (2.0 * np.pi)


def construct():
    t0 = time.time()
    deadline = t0 + 195.0

    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    seed_val = _value(seed)
    if seed_val is None:
        seed_val = 1.0 / (2.0 * np.pi)
    best_c = seed.copy()
    best_val = seed_val

    def _try_update(c):
        nonlocal best_c, best_val
        v = _value(c)
        if v is not None and v < best_val:
            best_c = c.copy()
            best_val = v

    starts = [seed.copy()]
    rng = np.random.default_rng(12345)
    for _ in range(30):
        starts.append(seed + rng.normal(0.0, 0.2, size=3))
    for s in [0.5, 1.0, 2.0, 3.0, 5.0]:
        starts.append(np.array([-0.1, s, 0.02]))
        starts.append(np.array([0.1, -s, 0.02]))

    for x0 in starts:
        if time.time() > deadline - 20.0:
            break
        try:
            res = minimize(_objective, x0, method="Nelder-Mead",
                           options={"maxiter": 1000, "xatol": 1e-10, "fatol": 1e-12,
                                    "adaptive": True})
            _try_update(res.x)
            if time.time() > deadline - 20.0:
                break
            try:
                res2 = minimize(_objective, res.x, method="L-BFGS-B",
                                options={"maxiter": 500, "gtol": 1e-14})
                _try_update(res2.x)
            except Exception:
                pass
        except Exception:
            continue

    if best_val >= seed_val:
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    return np.array(best_c, dtype=float)
# EVOLVE-BLOCK-END
