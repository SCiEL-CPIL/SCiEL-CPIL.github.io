"""Analytic uncertainty bound with precomputed constants + vectorized ops + multi-start L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200

def _precompute_constants():
    """Precompute all constant arrays needed for the bound calculation."""
    sp = np.sqrt(np.pi)
    
    # I[n] = integral x^{2n} e^{-x^2} dx
    # I[0] = sqrt(pi), I[n] = I[n-1] * (2n-1)/(2n)
    I = np.zeros(30)
    I[0] = sp
    for n in range(1, 30):
        I[n] = I[n-1] * (2*n - 1) / (2*n)
    
    # p coefficient matrix: p = p_mat @ [a0, a4, a8]
    # with a12 = -(a0 + 3a4 + 105a8)/945
    p_mat = np.zeros((7, 3))
    p_mat[0] = [1.0 - 1.0, 3.0 - 3.0, 105.0 - 105.0]  # 945*a12 cancels
    p_mat[1] = [7560.0/945.0, -6.0 + 7560.0*3.0/945.0, -420.0 + 7560.0*105.0/945.0]
    p_mat[2] = [-13860.0/945.0, 1.0 - 13860.0*3.0/945.0, 210.0 - 13860.0*105.0/945.0]
    p_mat[3] = [6930.0/945.0, 6930.0*3.0/945.0, -28.0 + 6930.0*105.0/945.0]
    p_mat[4] = [-1155.0/945.0, -1155.0*3.0/945.0, 1.0 - 1155.0*105.0/945.0]
    p_mat[5] = [66.0/945.0, 66.0*3.0/945.0, 66.0*105.0/945.0]
    p_mat[6] = [1.0/945.0, 3.0/945.0, 105.0/945.0]
    
    # q_k = p_k * (-1)^k
    sign = np.array([(-1.0)**k for k in range(7)])
    q_mat = p_mat * sign[np.newaxis, :]
    
    # dp matrix: dp[c, k] = dp_k/da_c
    dp_mat = np.zeros((3, 7))
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # dp0 = [1, 3, 105] + 945*da12
    dp_mat[0, 0] = 1.0 + 945.0*da12[0]
    dp_mat[1, 0] = 3.0 + 945.0*da12[1]
    dp_mat[2, 0] = 105.0 + 945.0*da12[2]
    
    # dp1 = [0, -6, -420] - 7560*da12
    dp_mat[0, 1] = 0.0 - 7560.0*da12[0]
    dp_mat[1, 1] = -6.0 - 7560.0*da12[1]
    dp_mat[2, 1] = -420.0 - 7560.0*da12[2]
    
    # dp2 = [0, 1, 210] + 13860*da12
    dp_mat[0, 2] = 0.0 + 13860.0*da12[0]
    dp_mat[1, 2] = 1.0 + 13860.0*da12[1]
    dp_mat[2, 2] = 210.0 + 13860.0*da12[2]
    
    # dp3 = [0, 0, -28] - 6930*da12
    dp_mat[0, 3] = 0.0 - 6930.0*da12[0]
    dp_mat[1, 3] = 0.0 - 6930.0*da12[1]
    dp_mat[2, 3] = -28.0 - 6930.0*da12[2]
    
    # dp4 = [0, 0, 1] + 1155*da12
    dp_mat[0, 4] = 0.0 + 1155.0*da12[0]
    dp_mat[1, 4] = 0.0 + 1155.0*da12[1]
    dp_mat[2, 4] = 1.0 + 1155.0*da12[2]
    
    # dp5 = [0, 0, 0] - 66*da12
    dp_mat[0, 5] = 0.0 - 66.0*da12[0]
    dp_mat[1, 5] = 0.0 - 66.0*da12[1]
    dp_mat[2, 5] = 0.0 - 66.0*da12[2]
    
    # dp6 = [0, 0, 0] + 1*da12
    dp_mat[0, 6] = 0.0 + 1.0*da12[0]
    dp_mat[1, 6] = 0.0 + 1.0*da12[1]
    dp_mat[2, 6] = 0.0 + 1.0*da12[2]
    
    return I, p_mat, q_mat, dp_mat, sign


# Precompute at module level
_I, _p_mat, _q_mat, _dp_mat, _sign = _precompute_constants()


def _bound_and_grad(params):
    """Compute uncertainty product and its analytic gradient using precomputed constants."""
    a0, a4, a8 = params[0], params[1], params[2]
    a_vec = np.array([a0, a4, a8])
    
    # Compute p and q vectors
    p = _p_mat @ a_vec
    q = _q_mat @ a_vec
    
    # Compute moments m_k = sum_{i,j} p_i p_j I[k+i+j] for k=0,1,2
    m = np.zeros(3)
    for k in range(3):
        # m_k = sum_i p_i * (sum_j p_j I[k+i+j])
        inner = np.zeros(7)
        for i in range(7):
            inner[i] = np.dot(p, _I[k+i:k+i+7])
        m[k] = np.dot(p, inner)
    
    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for k in range(3):
        inner = np.zeros(7)
        for i in range(7):
            inner[i] = np.dot(q, _I[k+i:k+i+7])
        n[k] = 2.0*np.pi * np.dot(q, inner)
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Compute gradient
    grad = np.zeros(3)
    for c in range(3):
        dp_c = _dp_mat[c, :]
        dq_c = dp_c * _sign
        
        # dm_k/da_c = 2 * sum_i dp_c[i] * (sum_j p_j I[k+i+j])
        dm = np.zeros(3)
        for k in range(3):
            s = np.zeros(7)
            for i in range(7):
                s[i] = np.dot(p, _I[k+i:k+i+7])
            dm[k] = 2.0 * np.dot(dp_c, s)
        
        # dn_k/da_c = 2*2*pi * sum_i dq_c[i] * (sum_j q_j I[k+i+j])
        dn = np.zeros(3)
        for k in range(3):
            s = np.zeros(7)
            for i in range(7):
                s[i] = np.dot(q, _I[k+i:k+i+7])
            dn[k] = 2.0 * 2.0*np.pi * np.dot(dq_c, s)
        
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
    # Multiple diverse starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([0.5, 0.01, 0.0001]),
        np.array([-0.8, 0.03, 0.0003]),
    ]
    
    rng = np.random.default_rng(123)
    for i in range(30):
        starts.append(rng.normal(0, 0.5, 3))
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 20:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish with Nelder-Mead from best
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish with L-BFGS-B
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
