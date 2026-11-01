"""Analytic uncertainty bound with improved multi-start optimization and better scaling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195  # safety margin before 210s kill


def _hermite_coeffs():
    """
    Precompute the polynomial coefficients of H_0, H_4, H_8, H_12.
    H_n are physicists' Hermite polynomials.
    H_0 = 1
    H_4 = x^4 - 6x^2 + 3
    H_8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    H_12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    
    We represent P(x) = sum_{k=0}^{6} p_k * x^{2k}
    """
    H = np.zeros((4, 7))
    H[0, 0] = 1.0
    H[1, 0] = 3.0; H[1, 1] = -6.0; H[1, 2] = 1.0
    H[2, 0] = 105.0; H[2, 1] = -420.0; H[2, 2] = 210.0; H[2, 3] = -28.0; H[2, 4] = 1.0
    H[3, 0] = 945.0; H[3, 1] = -7560.0; H[3, 2] = 13860.0; H[3, 3] = -6930.0
    H[3, 4] = 1155.0; H[3, 5] = -66.0; H[3, 6] = 1.0
    return H


def _compute_moments():
    """
    Precompute I[n] = integral x^{2n} e^{-x^2} dx = (2n-1)!! * sqrt(pi) / 2^n
    for n = 0, 1, ..., 12
    """
    I = np.zeros(13)
    sp = np.sqrt(np.pi)
    for n in range(13):
        if n == 0:
            I[n] = sp
        else:
            I[n] = I[n-1] * (2.0*n - 1.0) / (4.0 * n)
    return I


_H = _hermite_coeffs()
_I = _compute_moments()
_SIGN = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

# Precompute the matrix M where M[i,j] = I[i+j] for i,j in 0..6
_M0 = np.zeros((7, 7))
_M2 = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        _M0[i, j] = _I[i + j]
        _M2[i, j] = _I[1 + i + j]


def _bound(params):
    """
    Compute A(f)*A(F) for g(x) = P(x) * exp(-x^2/2)
    where P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12 with a12 = -(a0+3a4+105a8)/945
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H  # shape (7,)
    
    m0 = p @ _M0 @ p
    m2 = p @ _M2 @ p
    
    q = p * _SIGN
    n0 = 2.0 * np.pi * (q @ _M0 @ q)
    n2 = 2.0 * np.pi * (q @ _M2 @ q)
    
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18
    
    return np.sqrt(vf * vF)


def _bound_grad(params):
    """Compute bound and its gradient analytically using precomputed structures."""
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H
    
    m0 = p @ _M0 @ p
    m2 = p @ _M2 @ p
    
    q = p * _SIGN
    n0 = 2.0 * np.pi * (q @ _M0 @ q)
    n2 = 2.0 * np.pi * (q @ _M2 @ q)
    
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradient: d(bound)/da_c = (vF * d(vf)/da_c + vf * d(vF)/da_c) / (2*bound)
    # d(vf)/da_c = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    
    # da12/da_c for c = a0, a4, a8:
    da12 = np.array([-1.0 / 945.0, -3.0 / 945.0, -105.0 / 945.0])
    
    # Precompute row sums for efficiency
    # s0_i = sum_j p_j * I[i+j], s2_i = sum_j p_j * I[1+i+j]
    s0 = _M0 @ p
    s2 = _M2 @ p
    t0 = _M0 @ q
    t2 = _M2 @ q
    
    grad = np.zeros(3)
    for c in range(3):
        dp = _H[c, :] + da12[c] * _H[3, :]
        
        dm0 = 2.0 * (dp @ s0)
        dm2 = 2.0 * (dp @ s2)
        
        dq = dp * _SIGN
        dn0 = 4.0 * np.pi * (dq @ t0)
        dn2 = 4.0 * np.pi * (dq @ t2)
        
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
    Multi-start optimization with improved scaling and diverse exploration.
    """
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    # Scaling: normalize parameters to similar magnitude
    # a0 ~ O(1), a4 ~ O(0.04), a8 ~ O(0.0003)
    # Use scaled variables: b0 = a0, b1 = a4/0.04, b2 = a8/0.0003
    scale = np.array([1.0, 0.04, 0.0003])
    
    def _obj_scaled(b):
        a = b * scale
        return _bound(a)
    
    def _obj_grad_scaled(b):
        a = b * scale
        val, grad = _bound_grad(a)
        return val, grad / scale
    
    # Starting points in original space
    starts_orig = [
        np.array([-1.13, 0.04, 0.000308642]),   # seed
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.5, 0.03, 0.0002]),
        np.array([-0.8, 0.06, 0.0008]),
        np.array([-2.0, 0.02, 0.0001]),
        np.array([-1.2, 0.035, 0.0003]),
        np.array([-1.1, 0.045, 0.00035]),
        np.array([-0.5, 0.08, 0.001]),
        np.array([-3.0, 0.01, 0.00005]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -0.05, 0.001]),
        np.array([-2.5, 0.015, 0.00015]),
        np.array([-1.15, 0.042, 0.00032]),
        np.array([-1.05, 0.038, 0.00028]),
        np.array([-1.3, 0.03, 0.00025]),
        np.array([-0.9, 0.07, 0.0009]),
        np.array([-1.7, 0.02, 0.00015]),
        np.array([-1.4, 0.025, 0.0002]),
        np.array([-0.7, 0.09, 0.0012]),
    ]
    
    # Convert to scaled space
    starts = [s / scale for s in starts_orig]
    
    # Add diverse random starts with multiple seeds
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000]:
        rng = np.random.default_rng(seed)
        for _ in range(15):
            b = rng.normal([0, 0, 0], [1.0, 1.0, 1.0])
            starts.append(b)
    
    # Phase 1: L-BFGS-B from all starts (in scaled space)
    for start in starts:
        if time.time() > _DEADLINE - 25:
            break
        try:
            res = minimize(_obj_grad_scaled, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = (res.x * scale).copy()
        except Exception:
            pass
    
    # Phase 2: Polish with Nelder-Mead from best (in original space)
    if best_params is not None and time.time() < _DEADLINE - 20:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 100000, 'xatol': 1e-15, 'fatol': 1e-17})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final L-BFGS-B polish (in scaled space)
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            b_start = best_params / scale
            res = minimize(_obj_grad_scaled, b_start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-17, 'gtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = (res.x * scale).copy()
        except Exception:
            pass
    
    # Phase 4: Fine coordinate descent around best
    if best_params is not None and time.time() < _DEADLINE - 5:
        rng = np.random.default_rng(99999)
        for scale_cd in [0.1, 0.01, 0.001, 0.0001]:
            if time.time() > _DEADLINE - 3:
                break
            for _ in range(20):
                if time.time() > _DEADLINE - 3:
                    break
                perturbed = best_params + scale_cd * rng.normal(0, 1, 3)
                try:
                    b_pert = perturbed / scale
                    res = minimize(_obj_grad_scaled, b_pert, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 5000, 'ftol': 1e-17, 'gtol': 1e-16})
                    if res.fun < best_val and np.isfinite(res.fun):
                        best_val = res.fun
                        best_params = (res.x * scale).copy()
                except Exception:
                    pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
