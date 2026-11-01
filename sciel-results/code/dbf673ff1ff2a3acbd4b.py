"""Uncertainty inequality: global DE search + analytic gradient polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_T0 = time.time()
_DEADLINE = _T0 + 200.0  # Safety margin before 210s kill

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
_I = np.zeros(13)
for _n in range(13):
    log_val = 0.0
    for _m in range(1, 2 * _n + 1):
        log_val += np.log(_m)
    log_val -= _n * np.log(4.0)
    for _m in range(1, _n + 1):
        log_val -= np.log(_m)
    log_val += 0.5 * np.log(np.pi)
    _I[_n] = np.exp(log_val)

# Coefficient matrix: p = M @ [a0, a4, a8] after substituting a12 to enforce g(0)=0
_M = np.array([
    [0.0, 0.0, 0.0],
    [8.0, 18.0, 420.0],
    [-44.0 / 3.0, -43.0, -4410.0],
    [22.0 / 3.0, 22.0, 2282.0],
    [-11.0 / 9.0, -11.0 / 3.0, -10.0],
    [22.0 / 315.0, 22.0 / 105.0, 22.0 / 3.0],
    [-1.0 / 945.0, -1.0 / 315.0, -1.0 / 9.0],
])

_q_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
_Mq = _M * _q_signs[:, np.newaxis]


def _bound(params):
    """Compute the uncertainty bound A(f)*A(F)."""
    p = _M @ params
    q = _Mq @ params

    m0 = 0.0
    m1 = 0.0
    n0 = 0.0
    n1 = 0.0

    for i in range(7):
        pi = p[i]
        qi = q[i]
        for j in range(7):
            idx = i + j
            if idx < 13:
                m0 += pi * p[j] * _I[idx]
                n0 += 2.0 * np.pi * qi * q[j] * _I[idx]
            if idx + 1 < 13:
                m1 += pi * p[j] * _I[idx + 1]
                n1 += 2.0 * np.pi * qi * q[j] * _I[idx + 1]

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    vf = m1 / m0
    vF = n1 / n0
    if vf <= 0 or vF <= 0:
        return 1e18
    return np.sqrt(vf * vF)


def _bound_and_grad(params):
    """Compute bound and analytic gradient."""
    p = _M @ params
    q = _Mq @ params

    m0 = 0.0
    m1 = 0.0
    n0 = 0.0
    n1 = 0.0

    for i in range(7):
        pi = p[i]
        qi = q[i]
        for j in range(7):
            idx = i + j
            if idx < 13:
                m0 += pi * p[j] * _I[idx]
                n0 += 2.0 * np.pi * qi * q[j] * _I[idx]
            if idx + 1 < 13:
                m1 += pi * p[j] * _I[idx + 1]
                n1 += 2.0 * np.pi * qi * q[j] * _I[idx + 1]

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    vf = m1 / m0
    vF = n1 / n0
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    bound = np.sqrt(vf * vF)

    grad = np.zeros(3)
    for c in range(3):
        dp = _M[:, c]
        dq = _Mq[:, c]

        dm0 = 0.0
        dm1 = 0.0
        for i in range(7):
            dpi = dp[i]
            if dpi == 0:
                continue
            s0 = 0.0
            s1 = 0.0
            for j in range(7):
                idx0 = i + j
                if idx0 < 13:
                    s0 += p[j] * _I[idx0]
                if idx0 + 1 < 13:
                    s1 += p[j] * _I[idx0 + 1]
            dm0 += 2.0 * dpi * s0
            dm1 += 2.0 * dpi * s1

        dn0 = 0.0
        dn1 = 0.0
        for i in range(7):
            dqi = dq[i]
            if dqi == 0:
                continue
            s0 = 0.0
            s1 = 0.0
            for j in range(7):
                idx0 = i + j
                if idx0 < 13:
                    s0 += q[j] * _I[idx0]
                if idx0 + 1 < 13:
                    s1 += q[j] * _I[idx0 + 1]
            dn0 += 4.0 * np.pi * dqi * s0
            dn1 += 4.0 * np.pi * dqi * s1

        dvm2 = (dm1 * m0 - m1 * dm0) / (m0 * m0)
        dvn2 = (dn1 * n0 - n1 * dn0) / (n0 * n0)
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)

    return bound, grad


def _de_objective(params):
    """Objective for differential evolution (no gradient)."""
    val = _bound(params)
    return val


def construct():
    seed = np.array([-1.13, 0.04, 0.000308642])
    best_val = _bound(seed)
    best_params = seed.copy()

    # Phase 1: Differential evolution global search
    if time.time() < _DEADLINE - 60:
        try:
            bounds = [(-5.0, 5.0), (-2.0, 2.0), (-0.01, 0.01)]
            de_res = differential_evolution(
                _de_objective, bounds,
                seed=42, maxiter=200, popsize=20,
                tol=1e-10, mutation=(0.3, 1.2), recombination=0.8,
                polish=False, updating='deferred',
                init='sobol',
                workers=-1
            )
            if de_res.fun < best_val and np.isfinite(de_res.fun):
                best_val = de_res.fun
                best_params = de_res.x.copy()
        except Exception:
            pass

    # Phase 2: L-BFGS-B polish from best DE result
    if best_params is not None and time.time() < _DEADLINE - 20:
        try:
            res = minimize(_bound_and_grad, best_params, method='L-BFGS-B',
                           jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 3: Additional L-BFGS-B from multiple nearby starts
    if best_params is not None and time.time() < _DEADLINE - 10:
        rng = np.random.default_rng(7)
        for _ in range(8):
            if time.time() > _DEADLINE - 5:
                break
            perturbed = best_params + rng.normal(0, 0.05, 3)
            try:
                res = minimize(_bound_and_grad, perturbed, method='L-BFGS-B',
                               jac=True, options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-15})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass

    # Phase 4: Final Nelder-Mead polish
    if best_params is not None and time.time() < _DEADLINE - 3:
        try:
            res = minimize(_de_objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-17})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        return seed
    return np.array(best_params)
# EVOLVE-BLOCK-END
