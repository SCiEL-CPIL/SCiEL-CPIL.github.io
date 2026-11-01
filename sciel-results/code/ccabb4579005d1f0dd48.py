"""Uncertainty bound minimization with robust multi-stage optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 200


def _precompute():
    """Precompute I[n] = integral x^(2n) e^{-x^2} dx for n=0..13."""
    sp = np.sqrt(np.pi)
    I = np.zeros(14)
    I[0] = sp
    for n in range(1, 14):
        I[n] = I[n - 1] * (2 * n - 1) / (4 * n)
    return I


_I = _precompute()

# Coefficient matrices: P(x) = sum_{k=0}^{6} p_k x^{2k}
# p = M @ [a0, a4, a8, a12]
_M = np.array([
    [1.0,    3.0,   105.0,    945.0],
    [0.0,   -6.0,  -420.0,  -7560.0],
    [0.0,    1.0,   210.0,  13860.0],
    [0.0,    0.0,   -28.0,  -6930.0],
    [0.0,    0.0,     1.0,   1155.0],
    [0.0,    0.0,     0.0,    -66.0],
    [0.0,    0.0,     0.0,     1.0],
], dtype=np.float64)

_M3 = _M[:, :3]
_M12 = _M[:, 3:4]
_da12 = np.array([-1.0 / 945.0, -3.0 / 945.0, -105.0 / 945.0])

_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

# Gradient: dp_k/da_c
_dp_da = np.zeros((3, 7))
for c in range(3):
    _dp_da[c] = _M3[:, c] + _M12[0, 0] * _da12[c]

# A_k[i,j] = I[k+i+j]
_A = np.zeros((3, 7, 7))
for k in range(3):
    for i in range(7):
        for j in range(7):
            idx = k + i + j
            if idx < 14:
                _A[k, i, j] = _I[idx]

_Aq = np.zeros((3, 7, 7))
for k in range(3):
    for i in range(7):
        for j in range(7):
            _Aq[k, i, j] = _signs[i] * _A[k, i, j] * _signs[j]


def _bound_and_grad(params):
    """Compute bound and gradient. Returns (bound, grad)."""
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    p = _M3 @ params + _M12[0, 0] * a12

    # Check p is valid
    if not np.all(np.isfinite(p)):
        return 1e18, np.array([1e18, 1e18, 1e18])

    m0 = p @ _A[0] @ p
    m2 = p @ _A[1] @ p
    m4 = p @ _A[2] @ p

    n0 = 2.0 * np.pi * (p @ _Aq[0] @ p)
    n2 = 2.0 * np.pi * (p @ _Aq[1] @ p)
    n4 = 2.0 * np.pi * (p @ _Aq[2] @ p)

    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])

    bound = np.sqrt(vf * vF)

    if not np.isfinite(bound):
        return 1e18, np.array([1e18, 1e18, 1e18])

    # Gradient
    grad = np.zeros(3)
    for c in range(3):
        dp = _dp_da[c]

        r0 = _A[0] @ p
        r1 = _A[1] @ p
        r2 = _A[2] @ p

        dm0 = 2.0 * np.dot(dp, r0)
        dm2 = 2.0 * np.dot(dp, r1)
        dm4 = 2.0 * np.dot(dp, r2)

        dq = dp * _signs
        rq0 = _Aq[0] @ p
        rq1 = _Aq[1] @ p
        rq2 = _Aq[2] @ p

        dn0 = 4.0 * np.pi * np.dot(dq, rq0)
        dn2 = 4.0 * np.pi * np.dot(dq, rq1)
        dn4 = 4.0 * np.pi * np.dot(dq, rq2)

        dvm2 = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn2 = (dn2 * n0 - n2 * dn0) / (n0 * n0)

        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)

    if not np.all(np.isfinite(grad)):
        return bound, np.zeros(3)

    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val, _ = _bound_and_grad(seed_params)
    
    if not np.isfinite(seed_val):
        seed_val = 1e18
        seed_params = np.array([0.0, 0.0, 0.0])
        seed_val, _ = _bound_and_grad(seed_params)

    best_val = seed_val
    best_params = seed_params.copy()

    def _try_optimize(params, method, **kwargs):
        nonlocal best_val, best_params
        try:
            if time.time() > _DEADLINE - 15:
                return
            if method == 'L-BFGS-B':
                res = minimize(_objective_grad, params, method=method, jac=True, **kwargs)
            else:
                res = minimize(_objective, params, method=method, **kwargs)
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 1: Careful L-BFGS-B from seed
    _try_optimize(seed_params, 'L-BFGS-B',
                  options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16})

    # Phase 2: Multi-start with diverse points near seed
    if time.time() < _DEADLINE - 80:
        rng = np.random.default_rng(12345)
        # Small perturbations around seed
        for i in range(30):
            if time.time() > _DEADLINE - 15:
                break
            scale = 0.01 * (1.0 + i * 0.05)
            perturbation = rng.standard_normal(3) * scale
            perturbation[2] *= 0.01  # a8 is very small
            start = seed_params + perturbation
            _try_optimize(start, 'L-BFGS-B',
                          options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14})

    # Phase 3: Bounded differential evolution in tight region
    if time.time() < _DEADLINE - 50:
        try:
            # Tight bounds around seed
            bounds = [
                (seed_params[0] - 2.0, seed_params[0] + 2.0),
                (seed_params[1] - 0.1, seed_params[1] + 0.1),
                (seed_params[2] - 0.005, seed_params[2] + 0.005),
            ]
            de_result = differential_evolution(_objective, bounds, seed=777,
                                               maxiter=100, popsize=15,
                                               tol=1e-14, polish=True,
                                               init='sobol')
            if np.isfinite(de_result.fun) and de_result.fun < best_val:
                best_val = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass

    # Phase 4: Final high-precision polish
    if best_params is not None and time.time() < _DEADLINE - 10:
        _try_optimize(best_params, 'L-BFGS-B',
                      options={'maxiter': 20000, 'ftol': 1e-20, 'gtol': 1e-18})

    # Phase 5: Nelder-Mead final touch
    if best_params is not None and time.time() < _DEADLINE - 3:
        _try_optimize(best_params, 'Nelder-Mead',
                      options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18})

    # Safety: never return worse than seed
    if best_val > seed_val:
        best_params = seed_params
        best_val = seed_val

    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params, dtype=np.float64)
# EVOLVE-BLOCK-END
