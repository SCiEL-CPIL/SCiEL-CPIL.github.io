"""Optimized uncertainty bound via analytic Hermite-Gaussian moments + Dual Annealing + L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195  # 15s safety margin before 210s kill

# Precompute Gaussian moments G[m] = integral x^{2m} exp(-x^2) dx
_G_MOMENTS = []
def _init_gaussian_moments():
    global _G_MOMENTS
    if _G_MOMENTS:
        return
    _G_MOMENTS.append(np.sqrt(np.pi))
    for i in range(1, 20):
        _G_MOMENTS.append(_G_MOMENTS[-1] * (2 * i - 1) / 2.0)

_init_gaussian_moments()

def _gaussian_moment(m):
    if m < 0 or m >= len(_G_MOMENTS):
        # Fallback for safety, though max m in this problem is small
        if m < 0: return 0.0
        val = np.sqrt(np.pi)
        for i in range(1, m + 1):
            val *= (2 * i - 1) / 2.0
        return val
    return _G_MOMENTS[m]

def _compute_bound_and_grad(a0, a4, a8):
    """
    Compute the uncertainty bound and its analytic gradient.
    Returns (bound, grad).
    """
    # P(0) = a0 + 3a4 + 105a8 + 945a12 = 0 => a12 = -(a0 + 3a4 + 105a8)/945
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    # Coefficients of P(x) = sum p_k x^{2k}
    p = np.zeros(7)
    p[0] = 0.0
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p[2] = 1.0 * a4 + 210.0 * a8 + 13860.0 * a12
    p[3] = -28.0 * a8 - 6930.0 * a12
    p[4] = 1.0 * a8 + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = 1.0 * a12
    
    # Derivatives of p with respect to a0, a4, a8
    # dp/d(a0) = [0, 0, 0, 0, 0, 0, 0] ? No, a12 depends on a0.
    # a12' = [-1/945, -3/945, -105/945]
    # p_k depends on a12.
    # p_1 = -6 a4 - 420 a8 - 7560 a12
    # dp_1/da0 = -7560 * (-1/945) = 8
    # dp_1/da4 = -6 - 7560 * (-3/945) = -6 + 24 = 18
    # dp_1/da8 = -420 - 7560 * (-105/945) = -420 + 840 = 420
    
    # Let's compute dp/da matrix explicitly for safety and clarity
    # a12 = -1/945 a0 - 3/945 a4 - 105/945 a8
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    dp = np.zeros((7, 3))
    # p[1]
    dp[1] = np.array([-6.0, -420.0, -7560.0]) * da12 + np.array([0, -6.0, -420.0])
    # Wait, p[1] = -6 a4 - 420 a8 - 7560 a12
    # d/da0: -7560 * da12[0]
    # d/da4: -6 + -7560 * da12[1]
    # d/da8: -420 + -7560 * da12[2]
    dp[1, 0] = -7560.0 * da12[0]
    dp[1, 1] = -6.0 - 7560.0 * da12[1]
    dp[1, 2] = -420.0 - 7560.0 * da12[2]
    
    # p[2] = a4 + 210 a8 + 13860 a12
    dp[2, 0] = 13860.0 * da12[0]
    dp[2, 1] = 1.0 + 13860.0 * da12[1]
    dp[2, 2] = 210.0 + 13860.0 * da12[2]
    
    # p[3] = -28 a8 - 6930 a12
    dp[3, 0] = -6930.0 * da12[0]
    dp[3, 1] = -6930.0 * da12[1]
    dp[3, 2] = -28.0 - 6930.0 * da12[2]
    
    # p[4] = a8 + 1155 a12
    dp[4, 0] = 1155.0 * da12[0]
    dp[4, 1] = 1155.0 * da12[1]
    dp[4, 2] = 1.0 + 1155.0 * da12[2]
    
    # p[5] = -66 a12
    dp[5, 0] = -66.0 * da12[0]
    dp[5, 1] = -66.0 * da12[1]
    dp[5, 2] = -66.0 * da12[2]
    
    # p[6] = a12
    dp[6, 0] = da12[0]
    dp[6, 1] = da12[1]
    dp[6, 2] = da12[2]
    
    # M_0 = sum p_j p_k G[j+k]
    # M_2 = sum p_j p_k G[j+k+1]
    # N_2 = sum q_j q_k G[j+k+1]
    
    M_0 = 0.0
    M_2 = 0.0
    dM_0 = np.zeros(3)
    dM_2 = np.zeros(3)
    
    for j in range(7):
        if p[j] == 0:
            continue
        for k in range(7):
            val = p[j] * p[k]
            M_0 += val * _gaussian_moment(j + k)
            M_2 += val * _gaussian_moment(j + k + 1)
            # Gradient: d/da (p_j p_k) = dp_j/da p_k + p_j dp_k/da
            # dM_0 = sum (dp_j p_k + p_j dp_k) G[j+k]
            # Due to symmetry, we can just do 2 * sum_j p_j dp_j G[...] if we sum over all j,k?
            # Actually, dM_0 = 2 * sum_j p_j (sum_k dp_j/da * G[j+k])? No.
            # dM_0/da = sum_{j,k} (dp_j/da p_k + p_j dp_k/da) G[j+k]
            # = sum_{j,k} dp_j/da p_k G[j+k] + sum_{j,k} p_j dp_k/da G[j+k]
            # By swapping indices in the second term (j<->k), it is identical to the first.
            # So dM_0 = 2 * sum_{j,k} dp_j/da p_k G[j+k]
            
            for d in range(3):
                dM_0[d] += 2.0 * dp[j, d] * p[k] * _gaussian_moment(j + k)
                dM_2[d] += 2.0 * dp[j, d] * p[k] * _gaussian_moment(j + k + 1)
                
    # Q coefficients
    q = np.zeros(7)
    q[0] = 2.0 * p[1] - p[0]
    q[1] = 4.0 * p[2] - p[1]
    q[2] = 6.0 * p[3] - p[2]
    q[3] = 8.0 * p[4] - p[3]
    q[4] = 10.0 * p[5] - p[4]
    q[5] = 12.0 * p[6] - p[5]
    q[6] = -p[6]
    
    # dq/da
    dq = np.zeros((7, 3))
    for i in range(7):
        # q[i] is linear combination of p's
        # q[0] = 2 p[1] - p[0]
        # q[1] = 4 p[2] - p[1]
        # ...
        # q[6] = -p[6]
        if i == 0:
            dq[i] = 2.0 * dp[1] - dp[0]
        elif i == 6:
            dq[i] = -dp[6]
        else:
            dq[i] = (2 * (i + 1)) * dp[i + 1] - dp[i]
            
    N_2 = 0.0
    dN_2 = np.zeros(3)
    for j in range(7):
        if q[j] == 0:
            continue
        for k in range(7):
            val = q[j] * q[k]
            N_2 += val * _gaussian_moment(j + k + 1)
            for d in range(3):
                dN_2[d] += 2.0 * dq[j, d] * q[k] * _gaussian_moment(j + k + 1)
                
    if M_0 <= 0:
        return np.inf, np.zeros(3)
        
    bound = np.sqrt(M_2 * N_2) / M_0
    
    # Gradient of bound
    # J = (M_2 N_2)^0.5 M_0^-1
    # dJ = J * [ 0.5/M_2 dM_2 + 0.5/N_2 dN_2 - 1/M_0 dM_0 ]
    
    grad = np.zeros(3)
    if M_2 > 0 and N_2 > 0:
        factor = bound
        term_M2 = 0.5 * dM_2 / M_2
        term_N2 = 0.5 * dN_2 / N_2
        term_M0 = dM_0 / M_0
        grad = factor * (term_M2 + term_N2 - term_M0)
    else:
        grad = np.zeros(3)
        
    return bound, grad

def _objective(x):
    if time.time() > _DEADLINE:
        return np.inf
    try:
        val, _ = _compute_bound_and_grad(x[0], x[1], x[2])
        return val
    except:
        return np.inf

def _objective_grad(x):
    if time.time() > _DEADLINE:
        return np.zeros(3)
    try:
        _, grad = _compute_bound_and_grad(x[0], x[1], x[2])
        return grad
    except:
        return np.zeros(3)

def construct():
    # Start from the prior state of the art
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_x = x0.copy()
    best_val, _ = _compute_bound_and_grad(x0[0], x0[1], x0[2])
    
    # Use Dual Annealing for global search to escape local minima
    # Bounds are heuristic but wide
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    
    try:
        res_da = minimize(
            _objective,
            x0,
            method='Dual-Annealing',
            bounds=bounds,
            options={
                'maxiter': 100,
                'restarts': 20,
                'maxfun': 5000,
                'initial_temp': 5230.0,
                'visit': '2-tent',
                'seed': 42
            }
        )
        if res_da.fun < best_val:
            best_val = res_da.fun
            best_x = res_da.x.copy()
    except Exception:
        pass
    
    # Refine with L-BFGS-B using analytic gradient
    try:
        res_lbfgs = minimize(
            _objective,
            best_x,
            jac=_objective_grad,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 20000}
        )
        if res_lbfgs.fun < best_val:
            best_val = res_lbfgs.fun
            best_x = res_lbfgs.x.copy()
    except Exception:
        pass
        
    return best_x
# EVOLVE-BLOCK-END
