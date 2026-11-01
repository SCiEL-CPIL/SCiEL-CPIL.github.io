"""Structural change: coordinate-free ratio minimization with log-barrier interior-point + global restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 195

# Precompute I[n] = integral x^{2n} e^{-x^2} dx
def _compute_I_array(max_n=14):
    I = np.zeros(max_n + 1)
    I[0] = np.sqrt(np.pi)
    for n in range(1, max_n + 1):
        I[n] = I[n-1] * (2*n - 1) / (2*n)
    return I

_I = _compute_I_array(14)
_SP = np.sqrt(np.pi)
_TWO_PI = 2.0 * np.pi

# Hermite polynomial coefficients (physicist's convention)
# Basis: {1, x^2, x^4, x^6, x^8, x^10, x^12}
_H_COEFFS = np.array([
    [1, 0, 0, 0, 0, 0, 0],
    [3, -6, 1, 0, 0, 0, 0],
    [105, -420, 210, -28, 1, 0, 0],
    [945, -7560, 13860, -6930, 1155, -66, 1]
], dtype=float)

_NEG_ONE_POW = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

# Precompute the matrix M where m_k = p^T M_k p
# M_k[i,j] = I[k+i+j]
_M = np.zeros((3, 7, 7))
for k in range(3):
    for i in range(7):
        for j in range(7):
            idx = k + i + j
            if idx <= 14:
                _M[k, i, j] = _I[idx]

# For the Fourier side, Q_k = p_k * (-1)^k
# n_k = 2*pi * q^T M_k q where q = p * _NEG_ONE_POW
# This is equivalent to n_k = 2*pi * p^T D M_k D p where D = diag(_NEG_ONE_POW)
_D = np.diag(_NEG_ONE_POW)
_NM = np.zeros((3, 7, 7))
for k in range(3):
    _NM[k] = _TWO_PI * (_D @ _M[k] @ _D)


def _bound_and_grad(params, barrier_mu=0.0):
    """
    Compute sqrt(vf * vF) and gradient, optionally with log-barrier penalty.
    params = [a0, a4, a8]
    a12 = -(a0 + 3*a4 + 105*a8) / 945
    barrier_mu: if > 0, adds -mu * (log(m0) + log(n0)) to objective
    """
    a0, a4, a8 = params
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    coeffs = np.array([a0, a4, a8, a12])
    
    p = _H_COEFFS @ coeffs  # shape (7,)
    q = p * _NEG_ONE_POW
    
    # m_k = p^T M_k p
    m0 = p @ _M[0] @ p
    m1 = p @ _M[1] @ p
    n0 = q @ _NM[0] @ q
    n1 = q @ _NM[1] @ q
    
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m1 / m0
    vF = n1 / n0
    
    if vf <= 0 or vF <= 0 or not np.isfinite(vf) or not np.isfinite(vF):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    if not np.isfinite(bound):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    # Gradient computation
    da12 = np.array([-1.0, -3.0, -105.0]) / 945.0
    grad = np.zeros(3)
    
    for c in range(3):
        dp = _H_COEFFS[:, c] + da12[c] * _H_COEFFS[:, 3]
        dq = dp * _NEG_ONE_POW
        
        # dm_k = 2 * dp^T M_k p
        dm0 = 2.0 * dp @ _M[0] @ p
        dm1 = 2.0 * dp @ _M[1] @ p
        dn0 = 2.0 * dq @ _NM[0] @ q
        dn1 = 2.0 * dq @ _NM[1] @ q
        
        dvm = (dm1 * m0 - m1 * dm0) / (m0 * m0)
        dvn = (dn1 * n0 - n1 * dn0) / (n0 * n0)
        
        g_base = (vF * dvm + vf * dvn) / (2.0 * bound)
        
        # Log-barrier gradient: -mu * (dm0/m0 + dn0/n0)
        if barrier_mu > 0:
            g_barrier = -barrier_mu * (dm0 / m0 + dn0 / n0)
            grad[c] = g_base + g_barrier
        else:
            grad[c] = g_base
    
    obj_val = bound
    if barrier_mu > 0:
        obj_val += barrier_mu * (np.log(m0) + np.log(n0))
    
    return obj_val, grad


def _objective(params):
    val, _ = _bound_and_grad(params, barrier_mu=0.0)
    return val


def _objective_grad(params):
    return _bound_and_grad(params, barrier_mu=0.0)


def _barrier_objective(params, mu):
    val, _ = _bound_and_grad(params, barrier_mu=mu)
    return val


def _barrier_objective_grad(params, mu):
    return _bound_and_grad(params, barrier_mu=mu)


def construct():
    start_time = time.time()
    
    # Phase 1: Global optimization with differential evolution
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    if time.time() - start_time < _DEADLINE - 120:
        try:
            bounds = [(-10, 10), (-10, 10), (-10, 10)]
            de_result = differential_evolution(
                _objective, bounds,
                seed=42, maxiter=200, popsize=15,
                tol=1e-10, init='sobol',
                polish=True, options={'maxiter': 100}
            )
            if de_result.fun < best_val and np.isfinite(de_result.fun):
                best_val = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Multi-start L-BFGS-B with analytic gradient
    starts = [
        best_params.copy(),
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.5, 0.06, 0.0006]),
        np.array([-2.0, 0.08, 0.0008]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-2.5, 0.1, 0.001]),
        np.array([-3.0, 0.12, 0.0012]),
        np.array([0.5, -0.03, 0.0003]),
        np.array([-1.2, 0.045, 0.00035]),
        np.array([1.0, 0.05, -0.0005]),
        np.array([-0.8, -0.04, 0.0004]),
        np.array([2.0, -0.1, 0.001]),
        np.array([-4.0, 0.15, 0.0015]),
    ]
    
    rng = np.random.default_rng(123)
    for _ in range(10):
        starts.append(rng.normal(0, 1.5, 3))
    
    for start in starts:
        if time.time() - start_time > _DEADLINE - 60:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Log-barrier interior-point refinement
    # Use decreasing barrier parameter to guide toward the unconstrained minimum
    # while maintaining interior feasibility
    if best_params is not None and time.time() - start_time < _DEADLINE - 30:
        mu_schedule = [1.0, 0.1, 0.01, 0.001, 0.0001]
        current_params = best_params.copy()
        
        for mu in mu_schedule:
            if time.time() - start_time > _DEADLINE - 15:
                break
            try:
                # Create wrapper functions with fixed mu
                def barrier_obj(p, mu=mu):
                    return _barrier_objective(p, mu)
                
                def barrier_grad(p, mu=mu):
                    return _barrier_objective_grad(p, mu)
                
                res = minimize(barrier_grad, current_params, method='L-BFGS-B',
                              jac=True, options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-13})
                if np.isfinite(res.fun):
                    # Evaluate the true (unbarriered) objective at this point
                    true_val, _ = _bound_and_grad(res.x, barrier_mu=0.0)
                    if np.isfinite(true_val) and true_val < best_val:
                        best_val = true_val
                        best_params = res.x.copy()
                    current_params = res.x.copy()
            except Exception:
                pass
    
    # Phase 4: Trust-region refinement
    if best_params is not None and time.time() - start_time < _DEADLINE - 15:
        try:
            res = minimize(_objective_grad, best_params, method='trust-constr',
                          jac=True, options={'maxiter': 1000, 'gtol': 1e-12, 'xtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Nelder-Mead polish
    if best_params is not None and time.time() - start_time < _DEADLINE - 8:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 100000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 6: Final high-precision L-BFGS-B
    if best_params is not None and time.time() - start_time < _DEADLINE - 3:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        best_params = np.array([-1.13, 0.04, 0.000308642])
    
    return best_params
# EVOLVE-BLOCK-END
