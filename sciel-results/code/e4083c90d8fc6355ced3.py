"""Uncertainty bound minimization via vectorized analytic gradient,
constraint-free tangent-space reparametrization, and a high-precision polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195.0

# I[n] = integral x^{2n} exp(-x^2) dx, n = 0..12
_SP = np.sqrt(np.pi)
_I = np.zeros(13)
_I[0] = _SP
for _n in range(1, 13):
    _I[_n] = _I[_n - 1] * (2 * _n - 1) / 2.0

_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
_TWOPI = 2.0 * np.pi
_IX = np.arange(7)

# p_k (k=0..6) as a linear function of the free tangent coefficients
#   t = (a4, a8, a12),   a0 = -3 a4 - 105 a8 - 945 a12  (=> g(0)=0)
# p0 = -2 a4 - 210 a8
# p1 = -6 a4 - 420 a8 - 7560 a12
# p2 =   a4 + 210 a8 + 13860 a12
# p3 = -28 a8 - 6930 a12
# p4 =   a8 + 1155 a12
# p5 = -66 a12
# p6 =  a12
_DP = np.array([
    [-2.0,    -210.0,     0.0],
    [-6.0,    -420.0, -7560.0],
    [ 1.0,     210.0, 13860.0],
    [ 0.0,     -28.0, -6930.0],
    [ 0.0,       1.0,  1155.0],
    [ 0.0,       0.0,   -66.0],
    [ 0.0,       0.0,     1.0],
])


def _t_from_a(a0, a4, a8):
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    return np.array([a4, a8, a12])


def _bound_and_grad_t(t):
    """Bound and gradient w.r.t. t=(a4,a8,a12). Vectorized."""
    p = _DP @ t
    q = p * _SIGNS

    m0 = float(np.einsum('i,j,ij->', p, p, _I[_IX[:, None] + _IX[None, :]]))
    m1 = float(np.einsum('i,j,ij->', p, p, _I[_IX[:, None] + _IX[None, :] + 1]))
    n0 = float(np.einsum('i,j,ij->', q, q, _I[_IX[:, None] + _IX[None, :]]))
    n1 = float(np.einsum('i,j,ij->', q, q, _I[_IX[:, None] + _IX[None, :] + 1]))
    n0 *= _TWOPI
    n1 *= _TWOPI

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.zeros(3)

    vf = m1 / m0
    vF = n1 / n0
    if vf <= 0 or vF <= 0 or not np.isfinite(vf) or not np.isfinite(vF):
        return 1e18, np.zeros(3)

    bound = np.sqrt(vf * vF)

    # dm0/dt_c = 2 p . (dp_c . I)  ;  dp_c . I = I @ (_DP[:,c])
    S0 = _I @ _DP
    S1 = _I[1:] @ _DP
    pS0 = p @ S0
    pS1 = p @ S1
    qS0 = q @ S0
    qS1 = q @ S1

    dm0 = 2.0 * pS0
    dm1 = 2.0 * pS1
    dn0 = 2.0 * _TWOPI * qS0
    dn1 = 2.0 * _TWOPI * qS1

    dvf = (dm1 * m0 - m1 * dm0) / (m0 * m0)
    dvF = (dn1 * n0 - n1 * dn0) / (n0 * n0)

    grad = 0.5 / bound * (vF * dvf + vf * dvF)
    return bound, grad


def _objective_t(t):
    return _bound_and_grad_t(t)[0]


def _objective_grad_t(t):
    b, g = _bound_and_grad_t(t)
    if not np.isfinite(b):
        return 1e18, np.zeros(3)
    return b, g


def _coeffs_from_t(t):
    a4, a8, a12 = t
    a0 = -3.0 * a4 - 105.0 * a8 - 945.0 * a12
    return np.array([a0, a4, a8])


def construct():
    seed = np.array([-1.13, 0.04, 0.000308642])
    t_seed = _t_from_a(seed[0], seed[1], seed[2])

    # Verify seed validity
    sb, _ = _bound_and_grad_t(t_seed)
    if not np.isfinite(sb):
        return seed.copy()

    best_bound = sb
    best_t = t_seed.copy()

    rng = np.random.default_rng(12345)

    starts = [t_seed.copy()]

    # Perturbations around the seed (log-spaced scales)
    for scale in [0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0]:
        for _ in range(6):
            starts.append(t_seed + rng.normal(0.0, scale, 3))

    # Random points in a moderate box
    for _ in range(40):
        starts.append(rng.normal(0.0, 1.5, 3))

    # Axis / diagonal rays from the seed
    dirs = [
        np.array([1, 0, 0]), np.array([-1, 0, 0]),
        np.array([0, 1, 0]), np.array([0, -1, 0]),
        np.array([0, 0, 1]), np.array([0, 0, -1]),
        np.array([1, 1, 1]) / np.sqrt(3),
        np.array([-1, 1, 1]) / np.sqrt(3),
        np.array([1, -1, 1]) / np.sqrt(3),
        np.array([1, 1, -1]) / np.sqrt(3),
    ]
    for d in dirs:
        for r in [0.3, 0.6, 1.2, 2.5, 5.0]:
            starts.append(t_seed + r * d)

    # De-duplicate
    unique = [starts[0]]
    for s in starts[1:]:
        if all(np.linalg.norm(s - u) > 1e-7 for u in unique):
            unique.append(s)
    starts = unique

    options = {'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-11}

    for i, x0 in enumerate(starts):
        if time.time() > _DEADLINE - 25:
            break
        try:
            res = minimize(_objective_t, x0, method='L-BFGS-B',
                           jac=_objective_grad_t, options=options)
            b, _ = _bound_and_grad_t(res.x)
            if np.isfinite(b) and b < best_bound:
                best_bound = b
                best_t = res.x.copy()
        except Exception:
            continue

    # ---- High-precision polish on the best candidate ----
    if time.time() < _DEADLINE - 12:
        try:
            res = minimize(_objective_t, best_t, method='L-BFGS-B',
                           jac=_objective_grad_t,
                           options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14})
            b, _ = _bound_and_grad_t(res.x)
            if np.isfinite(b) and b < best_bound:
                best_bound = b
                best_t = res.x.copy()
        except Exception:
            pass

    if time.time() < _DEADLINE - 8:
        try:
            res = minimize(_objective_t, best_t, method='Nelder-Mead',
                           options={'maxiter': 40000, 'xatol': 1e-14,
                                    'fatol': 1e-16, 'adaptive': True})
            b, _ = _bound_and_grad_t(res.x)
            if np.isfinite(b) and b < best_bound:
                best_bound = b
                best_t = res.x.copy()
        except Exception:
            pass

    # ---- Final verification ----
    final, _ = _bound_and_grad_t(best_t)
    if not np.isfinite(final) or final > best_bound + 1e-9:
        return seed.copy()

    return _coeffs_from_t(best_t)
# EVOLVE-BLOCK-END
