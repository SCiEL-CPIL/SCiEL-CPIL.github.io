"""Analytic uncertainty bound with precomputed structure + differential evolution + local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 170  # safety margin before 210s kill


def _hermite_coeffs():
    """Precompute polynomial coefficients of H_0, H_4, H_8, H_12."""
    H = np.zeros((4, 7))
    H[0, 0] = 1.0
    H[1, 0] = 3.0; H[1, 1] = -6.0; H[1, 2] = 1.0
    H[2, 0] = 105.0; H[2, 1] = -420.0; H[2, 2] = 210.0; H[2, 3] = -28.0; H[2, 4] = 1.0
    H[3, 0] = 945.0; H[3, 1] = -7560.0; H[3, 2] = 13860.0; H[3, 3] = -6930.0
    H[3, 4] = 1155.0; H[3, 5] = -66.0; H[3, 6] = 1.0
    return H


def _compute_moments():
    """Precompute I[n] = integral x^{2n} e^{-x^2} dx for n = 0..13."""
    I = np.zeros(14)
    sp = np.sqrt(np.pi)
    for n in range(14):
        if n == 0:
            I[n] = sp
        else:
            I[n] = I[n-1] * (2.0*n - 1.0) / (4.0 * n)
    return I


_H = _hermite_coeffs()
_I = _compute_moments()
_SIGN = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

_idx = np.arange(7)
_I_mat0 = _I[_idx[:, None] + _idx[None, :]]
_I_mat2 = _I[1 + _idx[:, None] + _idx[None, :]]


def _bound(params):
    """Compute A(f)*A(F) for g(x) = P(x) * exp(-x^2/2)."""
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0

    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H

    m0 = p @ _I_mat0 @ p
    m2 = p @ _I_mat2 @ p

    q = p * _SIGN

    n0 = 2.0 * np.pi * (q @ _I_mat0 @ q)
    n2 = 2.0 * np.pi * (q @ _I_mat2 @ q)

    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0:
        return 1e18

    return np.sqrt(vf * vF)


def _bound_grad(params):
    """Compute bound and its analytic gradient using precomputed structures."""
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0

    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H

    m0 = p @ _I_mat0 @ p
    m2 = p @ _I_mat2 @ p

    q = p * _SIGN

    n0 = 2.0 * np.pi * (q @ _I_mat0 @ q)
    n2 = 2.0 * np.pi * (q @ _I_mat2 @ q)

    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])

    bound = np.sqrt(vf * vF)

    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

    grad = np.zeros(3)
    for c in range(3):
        dp = _H[c, :] + da12[c] * _H[3, :]

        dm0 = 2.0 * (dp @ _I_mat0 @ p)
        dm2 = 2.0 * (dp @ _I_mat2 @ p)

        dq = dp * _SIGN

        dn0 = 4.0 * np.pi * (dq @ _I_mat0 @ q)
        dn2 = 4.0 * np.pi * (dq @ _I_mat2 @ q)

        dvm = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn = (dn2 * n0 - n2 * dn0) / (n0 * n0)

        grad[c] = (vF * dvm + vf * dvn) / (2.0 * bound)

    return bound, grad


def _objective(params):
    return _bound(params)


def _objective_grad(params):
    return _bound_grad(params)


def construct():
    """
    Structural change: use differential evolution for global exploration,
    then multi-phase local polish. Also try a reparametrized search.
    """
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    # Evaluate seed
    seed_val = _bound(best_params)
    if seed_val < best_val:
        best_val = seed_val

    # Phase 1: Differential Evolution (global search)
    if time.time() < _DEADLINE - 60:
        try:
            bounds = [(-5.0, 5.0), (-0.5, 0.5), (-0.01, 0.01)]
            de_result = differential_evolution(
                _objective, bounds,
                seed=42, maxiter=200, popsize=30,
                tol=1e-12, mut=(0.5, 1.5), recomb=0.7,
                polish=False, init='latinhypercube',
                updating='immediate'
            )
            if de_result.fun < best_val and np.isfinite(de_result.fun):
                best_val = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass

    # Phase 2: L-BFGS-B from DE result + seed + diverse starts
    starts = [best_params.copy(), np.array([-1.13, 0.04, 0.000308642])]
    
    # Add structured starts around the best
    rng = np.random.default_rng(12345)
    for scale in [0.5, 0.1, 0.01]:
        starts.append(best_params + scale * np.array([-0.5, 0.02, 0.0002]))
        starts.append(best_params + scale * np.array([0.5, -0.02, 0.0002]))
        starts.append(best_params + scale * rng.normal(0, 1, 3))
    
    # Add diverse fixed starts
    diverse = [
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.5, 0.03, 0.0002]),
        np.array([-0.8, 0.06, 0.0008]),
        np.array([-2.0, 0.02, 0.0001]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -0.05, 0.001]),
        np.array([-2.5, 0.015, 0.00015]),
    ]
    starts.extend(diverse)
    
    # Random starts
    for _ in range(20):
        starts.append(rng.normal([best_params[0], 0.04, 0.0005], [1.5, 0.15, 0.003]))
    
    for start in starts:
        if time.time() > _DEADLINE - 40:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 3: Reparametrized search (log-scale for a4, a8)
    if time.time() < _DEADLINE - 35:
        def _obj_log(x):
            a0 = x[0]
            a4 = np.sign(x[1]) * np.exp(x[1] if abs(x[1]) > 1e-10 else 0.0)
            a4 = np.clip(a4, -0.5, 0.5)
            a8 = np.sign(x[2]) * np.exp(x[2] if abs(x[2]) > 1e-10 else 0.0)
            a8 = np.clip(a8, -0.01, 0.01)
            return _bound(np.array([a0, a4, a8]))
        
        def _obj_log_grad(x):
            a0 = x[0]
            a4 = np.sign(x[1]) * np.exp(x[1] if abs(x[1]) > 1e-10 else 0.0)
            a4 = np.clip(a4, -0.5, 0.5)
            a8 = np.sign(x[2]) * np.exp(x[2] if abs(x[2]) > 1e-10 else 0.0)
            a8 = np.clip(a8, -0.01, 0.01)
            val, grad = _bound_grad(np.array([a0, a4, a8]))
            # Chain rule for log transform
            g = np.zeros(3)
            g[0] = grad[0]
            if abs(x[1]) > 1e-10:
                g[1] = grad[1] * np.sign(x[1]) * np.exp(x[1])
            if abs(x[2]) > 1e-10:
                g[2] = grad[2] * np.sign(x[2]) * np.exp(x[2])
            return val, g
        
        # Convert best_params to log-space
        x0_log = np.array([best_params[0],
                           np.sign(best_params[1]) * np.log(abs(best_params[1]) + 1e-10),
                           np.sign(best_params[2]) * np.log(abs(best_params[2]) + 1e-10)])
        
        try:
            res = minimize(_obj_log_grad, x0_log, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
            a0 = res.x[0]
            a4 = np.sign(res.x[1]) * np.exp(res.x[1] if abs(res.x[1]) > 1e-10 else 0.0)
            a4 = np.clip(a4, -0.5, 0.5)
            a8 = np.sign(res.x[2]) * np.exp(res.x[2] if abs(res.x[2]) > 1e-10 else 0.0)
            a8 = np.clip(a8, -0.01, 0.01)
            p_new = np.array([a0, a4, a8])
            v_new = _bound(p_new)
            if v_new < best_val and np.isfinite(v_new):
                best_val = v_new
                best_params = p_new
        except Exception:
            pass

    # Phase 4: Nelder-Mead polish
    if best_params is not None and time.time() < _DEADLINE - 20:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-18, 'fatol': 1e-20, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 5: Trust-constr for high-precision
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective_grad, best_params, method='trust-constr',
                          jac=True, options={'maxiter': 3000, 'gtol': 1e-16, 'xtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 6: Final L-BFGS-B with tightest tolerances
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-18, 'gtol': 1e-17, 'maxls': 100})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 7: Perturbation refinement
    if best_params is not None and time.time() < _DEADLINE - 3:
        for scale in [0.01, 0.003, 0.001]:
            if time.time() > _DEADLINE - 1:
                break
            for _ in range(6):
                perturbed = best_params + scale * rng.normal(0, 1, 3)
                try:
                    res = minimize(_objective_grad, perturbed, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-17})
                    if res.fun < best_val and np.isfinite(res.fun):
                        best_val = res.fun
                        best_params = res.x.copy()
                except Exception:
                    pass

    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
