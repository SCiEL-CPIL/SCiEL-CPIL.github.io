"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
def _compute_I_table():
    sp = np.sqrt(np.pi)
    I = np.zeros(14)
    I[0] = sp
    for n in range(1, 14):
        I[n] = I[n-1] * (2*n - 1) / (4.0 * n)
    return I

_I_TABLE = _compute_I_table()

# Precompute coefficient matrices for P(x) in terms of (a0, a4, a8, a12)
# p0 = a0 + 3*a4 + 105*a8 + 945*a12
# p1 = -6*a4 - 420*a8 - 7560*a12
# p2 = a4 + 210*a8 + 13860*a12
# p3 = -28*a8 - 6930*a12
# p4 = a8 + 1155*a12
# p5 = -66*a12
# p6 = a12
_P_COEFFS = np.array([
    [1.0, 3.0, 105.0, 945.0],
    [0.0, -6.0, -420.0, -7560.0],
    [0.0, 1.0, 210.0, 13860.0],
    [0.0, 0.0, -28.0, -6930.0],
    [0.0, 0.0, 1.0, 1155.0],
    [0.0, 0.0, 0.0, -66.0],
    [0.0, 0.0, 0.0, 1.0]
])

# da12/da_c for c in {0,1,2}
_DA12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

# Signs for Fourier transform: q_k = p_k * (-1)^k
_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])


def _compute_p_vec(a0, a4, a8, a12):
    """Compute the 7-coefficient vector p for P(x)."""
    coeffs = np.array([a0, a4, a8, a12])
    return _P_COEFFS @ coeffs


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    p = _compute_p_vec(a0, a4, a8, a12)
    I = _I_TABLE
    
    # m_k = sum_{i,j=0}^{6} p_i p_j I[k+i+j] for k=0,1,2
    # Vectorized: m[k] = p^T M_k p where M_k[i,j] = I[k+i+j]
    m = np.zeros(3)
    for k in range(3):
        M_k = np.array([[I[k+i+j] for j in range(7)] for i in range(7)])
        m[k] = p @ M_k @ p
    
    # q_k = p_k * (-1)^k
    q = p * _SIGNS
    
    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for k in range(3):
        M_k = np.array([[I[k+i+j] for j in range(7)] for i in range(7)])
        n[k] = 2.0 * np.pi * (q @ M_k @ q)
    
    m0, m2, m4 = m[0], m[1], m[2]
    n0, n2, n4 = n[0], n[1], n[2]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Compute gradient analytically
    grad = np.zeros(3)
    for c in range(3):
        # dp/da_c = _P_COEFFS[:, c] + _P_COEFFS[:, 3] * _DA12[c]
        dp = _P_COEFFS[:, c] + _P_COEFFS[:, 3] * _DA12[c]
        
        # dm_k/da_c = 2 * sum_i dp_i * (sum_j p_j I[k+i+j])
        # = 2 * dp^T * (M_k @ p)
        dm = np.zeros(3)
        for k in range(3):
            M_k = np.array([[I[k+i+j] for j in range(7)] for i in range(7)])
            dm[k] = 2.0 * (dp @ (M_k @ p))
        
        dq = dp * _SIGNS
        
        dn = np.zeros(3)
        for k in range(3):
            M_k = np.array([[I[k+i+j] for j in range(7)] for i in range(7)])
            dn[k] = 2.0 * 2.0 * np.pi * (dq @ (M_k @ q))
        
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    # Multiple starting points - diverse coverage of parameter space
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.06, 0.0006]),
        np.array([-1.3, 0.03, 0.0003]),
        np.array([-2.5, 0.12, 0.0012]),
        np.array([0.5, -0.02, 0.0002]),
    ]
    
    rng = np.random.default_rng(123)
    for i in range(30):
        starts.append(rng.normal(0, 0.8, 3))
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 25:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 8000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish with Nelder-Mead from best
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 80000, 'xatol': 1e-17, 'fatol': 1e-19, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish with L-BFGS-B at tighter tolerance
    if best_params is not None and time.time() < _DEADLINE - 8:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-17, 'gtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # One more Nelder-Mead polish if time allows
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 30000, 'xatol': 1e-18, 'fatol': 1e-20, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
