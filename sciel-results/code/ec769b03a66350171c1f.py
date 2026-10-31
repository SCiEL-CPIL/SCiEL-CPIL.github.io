"""Analytic uncertainty bound with precomputed integrals + focused multi-start L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195

# Precompute I_n = integral x^{2n} e^{-x^2} dx for n=0..14
def _precompute_I():
    sp = np.sqrt(np.pi)
    I = np.zeros(15)
    I[0] = sp
    for n in range(1, 15):
        I[n] = I[n-1] * (2.0 * n - 1.0) / (2.0 * n)
    return I

_I = _precompute_I()

# Precompute dp_base and dp_a12
_DP_BASE = np.array([
    [1.0, 3.0, 105.0],   # p0
    [0.0, -6.0, -420.0], # p1
    [0.0, 1.0, 210.0],   # p2
    [0.0, 0.0, -28.0],   # p3
    [0.0, 0.0, 1.0],     # p4
    [0.0, 0.0, 0.0],     # p5
    [0.0, 0.0, 0.0],     # p6
])
_DP_A12 = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
_DA12 = np.array([-1.0 / 945.0, -3.0 / 945.0, -105.0 / 945.0])
_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])


def _bound_and_grad(params):
    """
    Compute uncertainty product sqrt(vf * vF) and its analytic gradient.
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    p = np.zeros(7)
    p[0] = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    p[3] = -28.0 * a8 - 6930.0 * a12
    p[4] = a8 + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = a12

    q = p * _SIGNS

    # m_k = sum_{i,j} p_i p_j I[k+i+j]
    m = np.zeros(3)
    for i in range(7):
        pi_val = p[i]
        if pi_val == 0.0:
            continue
        for j in range(7):
            pij = pi_val * p[j]
            if pij == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 15:
                    m[k] += pij * _I[idx]

    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for i in range(7):
        qi_val = q[i]
        if qi_val == 0.0:
            continue
        for j in range(7):
            qij = qi_val * q[j]
            if qij == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 15:
                    n[k] += 2.0 * np.pi * qij * _I[idx]

    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]

    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 1e-30 or vF <= 1e-30:
        return 1e18, np.array([1e18, 1e18, 1e18])

    bound = np.sqrt(vf * vF)

    # Gradient
    grad = np.zeros(3)
    for c in range(3):
        dp = _DP_BASE[:, c] + _DA12[c] * _DP_A12
        dq = dp * _SIGNS

        dm = np.zeros(3)
        for i in range(7):
            dpi = dp[i]
            if dpi == 0.0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 15:
                        s += p[j] * _I[idx]
                dm[k] += 2.0 * dpi * s

        dn = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if dqi == 0.0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 15:
                        s += q[j] * _I[idx]
                dn[k] += 4.0 * np.pi * dqi * s

        dvm2 = (dm[1] * m0 - m2 * dm[0]) / (m0 * m0)
        dvn2 = (dn[1] * n0 - n2 * dn[0]) / (n0 * n0)
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)

    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    # Focused set of starting points around the seed
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),   # seed
        np.array([-1.1, 0.04, 0.0003]),
        np.array([-1.15, 0.04, 0.00031]),
        np.array([-1.2, 0.045, 0.00035]),
        np.array([-1.0, 0.035, 0.00028]),
        np.array([-1.3, 0.05, 0.0004]),
        np.array([-0.9, 0.03, 0.00025]),
        np.array([-1.5, 0.055, 0.00045]),
        np.array([-0.8, 0.025, 0.0002]),
        np.array([-1.4, 0.05, 0.0004]),
    ]

    best_val = np.inf
    best_params = None

    for start in starts:
        if time.time() > _DEADLINE - 20:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # High-precision final polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 100})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])

    return np.array(best_params)
# EVOLVE-BLOCK-END
