"""Analytic uncertainty bound with precomputed constants + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200


def _precompute_constants():
    """Precompute all constant matrices and vectors needed for fast evaluation."""
    sp = np.sqrt(np.pi)
    
    # I[n] = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
    I = np.zeros(14)
    I[0] = sp
    for n in range(1, 14):
        I[n] = (2*n - 1) * I[n-1] / 2.0
    
    # p_k as linear functions of (a0, a4, a8) after substituting a12
    # a12 = -(a0 + 3*a4 + 105*a8) / 945
    P_mat = np.zeros((7, 3))
    # p0 = 0 (always, due to g(0)=0 constraint)
    P_mat[0, :] = [0.0, 0.0, 0.0]
    # p1 = 8*a0 + 18*a4 + 420*a8
    P_mat[1, :] = [8.0, 18.0, 420.0]
    # p2 = -(44/3)*a0 - 43*a4 - 1330*a8
    P_mat[2, :] = [-44.0/3.0, -43.0, -1330.0]
    # p3 = (22/3)*a0 + 22*a4 + 742*a8
    P_mat[3, :] = [22.0/3.0, 22.0, 742.0]
    # p4 = -(11/9)*a0 - 11*a4 - 112/9*a8
    P_mat[4, :] = [-11.0/9.0, -11.0, -112.0/9.0]
    # p5 = (22/315)*a0 + (22/105)*a4 + (22/3)*a8
    P_mat[5, :] = [22.0/315.0, 22.0/105.0, 22.0/3.0]
    # p6 = -(1/945)*a0 - (1/315)*a4 - (1/9)*a8
    P_mat[6, :] = [-1.0/945.0, -1.0/315.0, -1.0/9.0]
    
    # q_k = p_k * (-1)^k
    sign_vec = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    Q_mat = P_mat * sign_vec[:, np.newaxis]
    
    # Build moment matrices:
    # m_k = p^T M_k p where M_k[i,j] = I[k+i+j]
    M = np.zeros((3, 7, 7))
    for k in range(3):
        for i in range(7):
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    M[k, i, j] = I[idx]
    
    return P_mat, Q_mat, M


# Precompute at module level
_P_mat, _Q_mat, _M_mat = _precompute_constants()


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient using precomputed constants.
    """
    a0, a4, a8 = params[0], params[1], params[2]
    params3 = np.array([a0, a4, a8])
    
    # p = P_mat @ params3
    p = _P_mat @ params3
    
    # q = Q_mat @ params3
    q = _Q_mat @ params3
    
    # m_k = p^T M_k p
    m0 = p @ _M_mat[0] @ p
    m2 = p @ _M_mat[1] @ p
    
    # n_k = 2*pi * q^T M_k q
    n0 = 2.0 * np.pi * (q @ _M_mat[0] @ q)
    n2 = 2.0 * np.pi * (q @ _M_mat[1] @ q)
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradient:
    # d(bound)/d(a_c) = (1/(2*bound)) * (vF * d(vf)/d(a_c) + vf * d(vF)/d(a_c))
    # d(vf)/d(a_c) = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # dm_k/da_c = 2 * (M_k @ p) @ (P_mat[:, c])
    # d(vF)/d(a_c) = (dn2/da_c * n0 - n2 * dn0/da_c) / n0^2
    # dn_k/da_c = 2 * 2*pi * (M_k @ q) @ (Q_mat[:, c])
    
    grad = np.zeros(3)
    Mp0 = _M_mat[0] @ p
    Mp1 = _M_mat[1] @ p
    Mq0 = _M_mat[0] @ q
    Mq1 = _M_mat[1] @ q
    
    for c in range(3):
        dp = _P_mat[:, c]
        dq = _Q_mat[:, c]
        
        dm0 = 2.0 * (Mp0 @ dp)
        dm2 = 2.0 * (Mp1 @ dp)
        
        dn0 = 2.0 * 2.0 * np.pi * (Mq0 @ dq)
        dn2 = 2.0 * 2.0 * np.pi * (Mq1 @ dq)
        
        dvm2 = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn2 = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    # Multiple starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-1.2, 0.05, 0.001]),
        np.array([-1.0, 0.03, 0.0005]),
        np.array([-1.5, 0.06, 0.0006]),
        np.array([-0.8, 0.02, 0.0003]),
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
