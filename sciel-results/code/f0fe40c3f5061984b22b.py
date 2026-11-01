"""Uncertainty inequality: optimize Hermite-Gaussian coefficients to minimize C4 bound."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_T0 = time.time()
_DEADLINE = _T0 + 200.0  # Safety margin before 210s kill

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
_I = np.zeros(13)
for _n in range(13):
    if _n == 0:
        _I[_n] = np.sqrt(np.pi)
    else:
        _I[_n] = (2*_n - 1) / (2.0 * _n) * _I[_n - 1]

# Coefficient matrix: p_k = sum_c M[k,c] * a_c for c in {0,4,8}
# where a12 = -(a0 + 3*a4 + 105*a8)/945 enforces g(0)=0
_M = np.array([
    [0.0, 0.0, 0.0],
    [8.0, 18.0, 420.0],
    [-44.0/3.0, -43.0, -4410.0],
    [22.0/3.0, 22.0, 2282.0],
    [-11.0/9.0, -11.0/3.0, -10.0],
    [22.0/315.0, 22.0/105.0, 22.0/3.0],
    [-1.0/945.0, -1.0/315.0, -1.0/9.0],
])

# q_k = p_k * (-1)^k
_q_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
_Mq = _M * _q_signs[:, np.newaxis]


def _bound_and_grad(params):
    """Compute uncertainty product sqrt(<x^2>/<1> * <xi^2>/<1>) and its gradient."""
    a0, a4, a8 = params
    p = _M @ params
    q = _Mq @ params

    # m_k = sum_{i,j} p_i p_j I[k+i+j] for k=0,1,2
    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j] for k=0,1,2
    m = np.zeros(3)
    n = np.zeros(3)

    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0.0 and qi == 0.0:
            continue
        for j in range(7):
            idx0 = i + j
            if idx0 < 13:
                m[0] += pi * p[j] * _I[idx0]
                n[0] += 2.0 * np.pi * qi * q[j] * _I[idx0]
            idx1 = i + j + 1
            if idx1 < 13:
                m[1] += pi * p[j] * _I[idx1]
                n[1] += 2.0 * np.pi * qi * q[j] * _I[idx1]
            idx2 = i + j + 2
            if idx2 < 13:
                m[2] += pi * p[j] * _I[idx2]
                n[2] += 2.0 * np.pi * qi * q[j] * _I[idx2]

    m0, m2, m4 = m[0], m[1], m[2]
    n0, n2, n4 = n[0], n[1], n[2]

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0 or not np.isfinite(vf) or not np.isfinite(vF):
        return 1e18, np.array([1e18, 1e18, 1e18])

    bound = np.sqrt(vf * vF)

    # Analytic gradient
    grad = np.zeros(3)
    for c in range(3):
        dp = _M[:, c]
        dq = _Mq[:, c]

        dm = np.zeros(3)
        for i in range(7):
            dpi = dp[i]
            if dpi == 0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 13:
                        s += p[j] * _I[idx]
                dm[k] += 2.0 * dpi * s

        dn = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if dqi == 0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 13:
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
    global _DEADLINE
    _DEADLINE = time.time() + 200.0

    best_val = np.inf
    best_params = None

    # Phase 1: Differential evolution for global search
    bounds = [(-5.0, 5.0), (-2.0, 2.0), (-0.01, 0.01)]
    try:
        def _de_obj(x):
            if time.time() > _DEADLINE - 60:
                return 1e18
            return _objective(x)

        de_res = differential_evolution(_de_obj, bounds, seed=42, maxiter=200,
                                         tol=1e-14, polish=False, workers=1,
                                         init='sobol')
        if de_res.fun < best_val and np.isfinite(de_res.fun):
            best_val = de_res.fun
            best_params = de_res.x.copy()
    except Exception:
        pass

    # Phase 2: Multi-start L-BFGS-B from diverse points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-1.13, 0.04, 0.000308642]) + 0.01 * np.array([1.0, -1.0, 0.5]),
        np.array([-1.13, 0.04, 0.000308642]) - 0.01 * np.array([1.0, -1.0, 0.5]),
    ]

    rng = np.random.default_rng(123)
    for i in range(20):
        starts.append(rng.normal(0, 0.5, 3))

    if best_params is not None:
        starts.append(best_params)
        starts.append(best_params + 0.05 * rng.normal(0, 1, 3))
        starts.append(best_params - 0.05 * rng.normal(0, 1, 3))

    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                           jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 3: Nelder-Mead polish
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 100000, 'xatol': 1e-18, 'fatol': 1e-20})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 4: Final L-BFGS-B polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                           jac=True, options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])

    return np.array(best_params)
