"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 200


def _precompute_I():
    """Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)"""
    sp = np.sqrt(np.pi)
    I = np.zeros(14)
    for n in range(14):
        if n == 0:
            I[n] = sp
        else:
            # (2n)! / (4^n n!) * sqrt(pi)
            # Use recurrence: I_n = I_{n-1} * (2n-1) / (4n)
            I[n] = I[n-1] * (2*n - 1) / (4*n)
    return I

_I_TABLE = _precompute_I()

# Precompute the Jacobian-like matrices for gradient computation
# dp/da_c for c=0,1,2 (coefficients a0, a4, a8)
# p0 = a0+3a4+105a8+945a12, p1 = -6a4-420a8-7560a12, etc.
# a12 = -(a0+3a4+105a8)/945
# da12/da0=-1/945, da12/da4=-3/945, da12/da8=-105/945

_DP = np.zeros((3, 7))
for c in range(3):
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])[c]
    _DP[c, 0] = np.array([1.0, 3.0, 105.0])[c] + 945.0*da12
    _DP[c, 1] = np.array([0.0, -6.0, -420.0])[c] - 7560.0*da12
    _DP[c, 2] = np.array([0.0, 1.0, 210.0])[c] + 13860.0*da12
    _DP[c, 3] = np.array([0.0, 0.0, -28.0])[c] - 6930.0*da12
    _DP[c, 4] = np.array([0.0, 0.0, 1.0])[c] + 1155.0*da12
    _DP[c, 5] = np.array([0.0, 0.0, 0.0])[c] - 66.0*da12
    _DP[c, 6] = np.array([0.0, 0.0, 0.0])[c] + 1.0*da12

# Precompute sign factors for Fourier transform
_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])  # (-1)^k for k=0..6


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    p = np.zeros(7)
    p[0] = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    
    I = _I_TABLE
    
    # m_k = sum_{i,j=0}^{6} p_i p_j I[k+i+j] for k=0,1,2
    # Use matrix form: m = P^T * M * P where M[k,i,j] = I[k+i+j]
    # But simpler: compute via outer product
    
    # m0 = sum_{i,j} p_i p_j I[i+j]
    # m1 = sum_{i,j} p_i p_j I[1+i+j]
    # m2 = sum_{i,j} p_i p_j I[2+i+j]
    
    # Precompute sum_j p_j I[i+j] for each i
    s_p = np.zeros(7)  # s_p[i] = sum_j p_j I[i+j]
    for i in range(7):
        s_p[i] = np.dot(p, I[i:i+7])
    
    m0 = np.dot(p, s_p)
    m1 = np.dot(p, np.array([np.dot(p, I[i+1:i+8]) for i in range(7)]))
    m2 = np.dot(p, np.array([np.dot(p, I[i+2:i+9]) for i in range(7)]))
    
    # For Fourier: q_k = p_k * (-1)^k
    q = p * _SIGNS
    
    s_q = np.zeros(7)
    for i in range(7):
        s_q[i] = np.dot(q, I[i:i+7])
    
    n0 = 2.0 * np.pi * np.dot(q, s_q)
    n1 = 2.0 * np.pi * np.dot(q, np.array([np.dot(q, I[i+1:i+8]) for i in range(7)]))
    n2 = 2.0 * np.pi * np.dot(q, np.array([np.dot(q, I[i+2:i+9]) for i in range(7)]))
    
    m0, m2 = m0, m2
    n0, n2 = n0, n2
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Compute gradient
    # d(bound)/d(a_c) = (1/(2*bound)) * (vF * d(vf)/d(a_c) + vf * d(vF)/d(a_c))
    # d(vf)/d(a_c) = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # d(vF)/d(a_c) = (dn2/da_c * n0 - n2 * dn0/da_c) / n0^2
    
    grad = np.zeros(3)
    for c in range(3):
        dp = _DP[c]
        dq = dp * _SIGNS
        
        # dm_k/da_c = 2 * sum_i dp_i * (sum_j p_j I[k+i+j])
        dm0 = 2.0 * np.dot(dp, s_p)
        dm1 = 2.0 * np.dot(dp, np.array([np.dot(p, I[i+1:i+8]) for i in range(7)]))
        dm2 = 2.0 * np.dot(dp, np.array([np.dot(p, I[i+2:i+9]) for i in range(7)]))
        
        dn0 = 2.0 * 2.0 * np.pi * np.dot(dq, s_q)
        dn1 = 2.0 * 2.0 * np.pi * np.dot(dq, np.array([np.dot(q, I[i+1:i+8]) for i in range(7)]))
        dn2 = 2.0 * 2.0 * np.pi * np.dot(dq, np.array([np.dot(q, I[i+2:i+9]) for i in range(7)]))
        
        dvm2 = (dm2*m0 - m2*dm0) / (m0*m0)
        dvn2 = (dn2*n0 - n2*dn0) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    # Multiple starting points - more diverse
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.06, 0.0005]),
        np.array([-1.2, 0.03, 0.00025]),
        np.array([-1.8, 0.12, 0.001]),
        np.array([0.5, -0.03, 0.0003]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(30):
        starts.append(rng.normal(0, 0.8, 3))
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 25:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 8000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish with Nelder-Mead from best
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 80000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish with L-BFGS-B
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
