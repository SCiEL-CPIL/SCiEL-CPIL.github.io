"""
Analytic uncertainty bound minimization via exact gradient + multi-start L-BFGS-B
with Nelder-Mead polish and a coarse-to-fine grid fallback.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195  # safety margin before 210s kill


def _hermite_coeffs():
    """
    Precompute the polynomial coefficients of H_0, H_4, H_8, H_12.
    Represent P(x) = sum_{k=0}^{6} p_k * x^{2k}
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
    for n = 0, 1, ..., 13
    """
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


def _bound_and_grad(params):
    """
    Compute the uncertainty bound A(f)*A(F) and its analytic gradient.
    g(x) = P(x) * exp(-x^2/2)
    P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12 with a12 = -(a0+3a4+105a8)/945
    
    Returns (bound, grad) where grad is d(bound)/d(a0,a4,a8)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H  # shape (7,)
    
    # m_k = int x^{2k} |P|^2 e^{-x^2} dx = sum_{i,j=0}^{6} p_i p_j I[k+i+j]
    m0 = 0.0
    m2 = 0.0
    for i in range(7):
        pi = p[i]
        if pi == 0.0:
            continue
        for j in range(7):
            pij = pi * p[j]
            m0 += pij * _I[i+j]
            m2 += pij * _I[1+i+j]
    
    # Q(xi) = P(-i*xi): q_k = p_k * (-1)^k
    q = p * _SIGN
    
    # n_k = 2*pi * int xi^{2k} |Q|^2 e^{-xi^2} dx
    n0 = 0.0
    n2 = 0.0
    for i in range(7):
        qi = q[i]
        if qi == 0.0:
            continue
        for j in range(7):
            qij = qi * q[j]
            n0 += 2.0 * np.pi * qij * _I[i+j]
            n2 += 2.0 * np.pi * qij * _I[1+i+j]
    
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradient computation
    # d(bound)/da_c = (vF * d(vf)/da_c + vf * d(vF)/da_c) / (2*bound)
    # d(vf)/da_c = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    grad = np.zeros(3)
    for c in range(3):
        dp = _H[c, :] + da12[c] * _H[3, :]
        
        dm0 = 0.0
        dm2 = 0.0
        for i in range(7):
            dpi = dp[i]
            if dpi == 0.0:
                continue
            s0 = 0.0
            s2 = 0.0
            for j in range(7):
                s0 += p[j] * _I[i+j]
                s2 += p[j] * _I[1+i+j]
            dm0 += 2.0 * dpi * s0
            dm2 += 2.0 * dpi * s2
        
        dq = dp * _SIGN
        
        dn0 = 0.0
        dn2 = 0.0
        for i in range(7):
            dqi = dq[i]
            if dqi == 0.0:
                continue
            s0 = 0.0
            s2 = 0.0
            for j in range(7):
                s0 += q[j] * _I[i+j]
                s2 += q[j] * _I[1+i+j]
            dn0 += 4.0 * np.pi * dqi * s0
            dn2 += 4.0 * np.pi * dqi * s2
        
        dvm = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        
        grad[c] = (vF * dvm + vf * dvn) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    return _bound_and_grad(params)


def construct():
    """
    Multi-start L-BFGS-B with Nelder-Mead polish and coarse-to-fine grid fallback.
    Returns the three free coefficients (a0, a4, a8).
    """
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    # Phase 1: Coarse grid search to find a good basin
    center = np.array([-1.0, 0.0, 0.0])
    half_widths = np.array([2.5, 1.5, 0.8])
    n_points = np.array([15, 12, 8])
    
    a0_vals = np.linspace(center[0] - half_widths[0], center[0] + half_widths[0], n_points[0])
    a4_vals = np.linspace(center[1] - half_widths[1], center[1] + half_widths[1], n_points[1])
    a8_vals = np.linspace(center[2] - half_widths[2], center[2] + half_widths[2], n_points[2])
    
    for a0 in a0_vals:
        if time.time() > _DEADLINE - 30:
            break
        for a4 in a4_vals:
            if time.time() > _DEADLINE - 30:
                break
            for a8 in a8_vals:
                if time.time() > _DEADLINE - 30:
                    break
                params = np.array([a0, a4, a8])
                try:
                    val, _ = _bound_and_grad(params)
                    if val < best_val and np.isfinite(val):
                        best_val = val
                        best_params = params.copy()
                except Exception:
                    pass
    
    # Phase 2: Multi-start L-BFGS-B from diverse starting points
    starts = [
        best_params.copy(),
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.2, 0.03, 0.0002]),
        np.array([-0.8, 0.06, 0.0005]),
        np.array([-1.5, 0.0, 0.0]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(15):
        starts.append(rng.normal(0, 0.8, 3))
    
    for start in starts:
        if time.time() > _DEADLINE - 15:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-25, 'gtol': 1e-22, 'maxls': 500})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead polish from the best point
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 100000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision L-BFGS-B polish
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 50000, 'ftol': 1e-30, 'gtol': 1e-25, 'maxls': 1000})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
