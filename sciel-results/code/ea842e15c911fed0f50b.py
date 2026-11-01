"""Analytic uncertainty bound with exact gradient + Log-Barrier Interior Point Optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195

# Precompute integration constants I_n = integral x^{2n} e^{-x^2} dx
# I_0 = sqrt(pi)
# I_n = (2n-1)/2 * I_{n-1}
_I = np.zeros(14)
_I[0] = np.sqrt(np.pi)
for n in range(1, 14):
    _I[n] = (2 * n - 1) / 2.0 * _I[n - 1]

# Precompute matrices M_k for moments m_k = p^T M_k p
# p has length 7 (p_0 to p_6)
# m_k = sum_{i,j} p_i p_j I_{k+i+j}
# M_k[i,j] = I_{k+i+j} if k+i+j < 14 else 0
_M = np.zeros((3, 7, 7))
for k in range(3):
    for i in range(7):
        for j in range(7):
            idx = k + i + j
            if idx < 14:
                _M[k, i, j] = _I[idx]

# Similarly for Fourier moments n_k
# Q(x) = sum q_k x^{2k} where q_k = p_k (-1)^k
# n_k = 2*pi * sum_{i,j} q_i q_j I_{k+i+j}
# We can express this in terms of p.
# q = D p, where D = diag(1, -1, 1, -1, 1, -1, 1)
_D = np.diag([1, -1, 1, -1, 1, -1, 1])
# N_k = 2*pi * D M_k D
_N = np.zeros((3, 7, 7))
for k in range(3):
    _N[k] = 2.0 * np.pi * (_D @ _M[k] @ _D)


def _coeffs_to_p(a0, a4, a8):
    """
    Compute polynomial coefficients p[0..6] from a0, a4, a8.
    a12 = -(a0 + 3*a4 + 105*a8) / 945
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    p = np.zeros(7)
    # p0
    p[0] = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    # p1
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    # p2
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    # p3
    p[3] = -28.0 * a8 - 6930.0 * a12
    # p4
    p[4] = a8 + 1155.0 * a12
    # p5
    p[5] = -66.0 * a12
    # p6
    p[6] = a12
    return p


def _bound_and_grad_2d(params):
    """
    params: [u, v] where u = a4/a0, v = a8/a0.
    We set a0 = 1 for calculation.
    Returns: bound, grad (2D)
    """
    u, v = params[0], params[1]
    a0 = 1.0
    a4 = u
    a8 = v
    
    p = _coeffs_to_p(a0, a4, a8)
    
    # Moments
    # m_k = p^T M_k p
    m0 = p @ _M[0] @ p
    m2 = p @ _M[1] @ p
    # m4 not needed for bound, only m0, m2
    
    # n_k = p^T N_k p
    n0 = p @ _N[0] @ p
    n2 = p @ _N[1] @ p
    
    if m0 <= 1e-12 or n0 <= 1e-12 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18])
        
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 1e-12 or vF <= 1e-12:
        return 1e18, np.array([1e18, 1e18])
        
    bound = np.sqrt(vf * vF)
    
    # Gradient
    # d(bound)/d(u) = (1/(2*bound)) * (vF * d(vf)/du + vf * d(vF)/du)
    # d(vf)/du = (dm2/du * m0 - m2 * dm0/du) / m0^2
    # dm_k/du = 2 * (M_k p) * (dp/du)
    
    # dp/du and dp/dv
    # p depends on a12 which depends on a0, a4, a8.
    # a0=1 fixed.
    # a4 = u, a8 = v.
    # a12 = -(1 + 3u + 105v)/945
    
    da12_du = -3.0 / 945.0
    da12_dv = -105.0 / 945.0
    
    # dp/du
    dp_du = np.zeros(7)
    dp_du[0] = 3.0 + 945.0 * da12_du
    dp_du[1] = -6.0 - 7560.0 * da12_du
    dp_du[2] = 1.0 + 13860.0 * da12_du
    dp_du[3] = -6930.0 * da12_du
    dp_du[4] = 1155.0 * da12_du
    dp_du[5] = -66.0 * da12_du
    dp_du[6] = da12_du
    
    # dp/dv
    dp_dv = np.zeros(7)
    dp_dv[0] = 105.0 + 945.0 * da12_dv
    dp_dv[1] = -420.0 - 7560.0 * da12_dv
    dp_dv[2] = 210.0 + 13860.0 * da12_dv
    dp_dv[3] = -28.0 - 6930.0 * da12_dv
    dp_dv[4] = 1.0 + 1155.0 * da12_dv
    dp_dv[5] = -66.0 * da12_dv
    dp_dv[6] = da12_dv
    
    grad = np.zeros(2)
    
    for i, dp in enumerate([dp_du, dp_dv]):
        # dm0 = 2 * (M0 p) dot dp
        # dm2 = 2 * (M1 p) dot dp
        # Note: M_k are symmetric.
        
        Mp0 = _M[0] @ p
        Mp1 = _M[1] @ p
        
        dm0 = 2.0 * (Mp0 @ dp)
        dm2 = 2.0 * (Mp1 @ dp)
        
        Np0 = _N[0] @ p
        Np1 = _N[1] @ p
        
        dn0 = 2.0 * (Np0 @ dp)
        dn2 = 2.0 * (Np1 @ dp)
        
        dvm2 = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn2 = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        
        grad[i] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
        
    return bound, grad


def _objective_barrier(params, epsilon):
    val, grad = _bound_and_grad_2d(params)
    if val > 1e17:
        return val, grad
        
    # Add log barrier
    # We need V_x and V_xi to compute log
    u, v = params[0], params[1]
    a0 = 1.0
    a4 = u
    a8 = v
    p = _coeffs_to_p(a0, a4, a8)
    m0 = p @ _M[0] @ p
    m2 = p @ _M[1] @ p
    n0 = p @ _N[0] @ p
    n2 = p @ _N[1] @ p
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18])
        
    barrier = -epsilon * (np.log(vf) + np.log(vF))
    
    # Gradient of barrier
    # d/dp ( -eps * ln(vf) ) = -eps * (1/vf) * d(vf)/dp
    # We already have d(vf)/du in grad calculation logic, but let's recompute for clarity or reuse.
    # Reusing is safer to avoid code duplication errors.
    # Let's just compute the barrier gradient numerically? No, analytic is better.
    # Actually, the previous function returns grad of bound.
    # grad_barrier = -eps * (1/vf * d(vf)/dparams + 1/vF * d(vF)/dparams)
    # This is exactly -eps * (vF * d(vf) + vf * d(vF)) / (vf * vF) * (1/2 * bound)? No.
    # d(bound) = 1/(2*bound) * (vF d(vf) + vf d(vF))
    # d(barrier) = -eps * (d(vf)/vf + d(vF)/vF)
    # Let T = vF d(vf) + vf d(vF).
    # d(bound) = T / (2 * bound)
    # d(barrier) = -eps * ( (1/vf) d(vf) + (1/vF) d(vF) )
    # This is not directly proportional to T.
    # Let's compute d(vf) and d(vF) separately.
    
    # To save time, I will just use the bound gradient and add a small numerical gradient for the barrier if epsilon is small, 
    # or just compute the full analytic gradient for the barrier term.
    
    # Let's refactor _bound_and_grad_2d to return dvm2 and dvn2 as well? 
    # Or just compute the barrier gradient analytically here.
    
    # It's easier to just compute the full gradient of (bound + barrier) in one go.
    # I'll create a helper that returns bound, dvm2, dvn2.
    
    return val + barrier, grad # Placeholder, will fix in final code

def _full_obj_grad(params, epsilon):
    """
    Computes objective and gradient for:
    J = sqrt(vf * vF) - epsilon * (ln(vf) + ln(vF))
    """
    u, v = params[0], params[1]
    a0 = 1.0
    a4 = u
    a8 = v
    
    p = _coeffs_to_p(a0, a4, a8)
    
    m0 = p @ _M[0] @ p
    m2 = p @ _M[1] @ p
    n0 = p @ _N[0] @ p
    n2 = p @ _N[1] @ p
    
    if m0 <= 1e-12 or n0 <= 1e-12:
        return 1e18, np.array([1e18, 1e18])
        
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 1e-12 or vF <= 1e-12:
        return 1e18, np.array([1e18, 1e18])
        
    bound = np.sqrt(vf * vF)
    barrier_val = -epsilon * (np.log(vf) + np.log(vF))
    obj_val = bound + barrier_val
    
    # Gradients
    da12_du = -3.0 / 945.0
    da12_dv = -105.0 / 945.0
    
    dp_du = np.array([
        3.0 + 945.0 * da12_du,
        -6.0 - 7560.0 * da12_du,
        1.0 + 13860.0 * da12_du,
        -6930.0 * da12_du,
        1155.0 * da12_du,
        -66.0 * da12_du,
        da12_du
    ])
    
    dp_dv = np.array([
        105.0 + 945.0 * da12_dv,
        -420.0 - 7560.0 * da12_dv,
        210.0 + 13860.0 * da12_dv,
        -28.0 - 6930.0 * da12_dv,
        1.0 + 1155.0 * da12_dv,
        -66.0 * da12_dv,
        da12_dv
    ])
    
    grad = np.zeros(2)
    
    Mp0 = _M[0] @ p
    Mp1 = _M[1] @ p
    Np0 = _N[0] @ p
    Np1 = _N[1] @ p
    
    for i, dp in enumerate([dp_du, dp_dv]):
        dm0 = 2.0 * (Mp0 @ dp)
        dm2 = 2.0 * (Mp1 @ dp)
        dn0 = 2.0 * (Np0 @ dp)
        dn2 = 2.0 * (Np1 @ dp)
        
        dvm2 = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn2 = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        
        # Gradient of bound
        g_bound = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
        
        # Gradient of barrier: -eps * (dvm2/vf + dvn2/vF)
        g_barrier = -epsilon * (dvm2 / vf + dvn2 / vF)
        
        grad[i] = g_bound + g_barrier
        
    return obj_val, grad


def construct():
    best_val = np.inf
    best_params = None
    
    # Multi-start in 2D space (u, v)
    # Seed: a0=-1.13, a4=0.04, a8=0.000308642
    # u = a4/a0 = -0.035398, v = a8/a0 = -0.00027313
    starts = [
        np.array([-0.035398, -0.00027313]),
        np.array([0.0, 0.0]),
        np.array([-0.05, 0.001]),
        np.array([-0.02, -0.001]),
        np.array([0.05, 0.005]),
        np.array([-0.1, 0.01]),
    ]
    
    rng = np.random.default_rng(123)
    for i in range(10):
        starts.append(rng.normal(0, 0.1, 2))
    
    # Stages of epsilon
    epsilons = [1.0, 0.1, 0.01, 0.0]
    
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
            
        current_params = start
        for eps in epsilons:
            if time.time() > _DEADLINE - 10:
                break
            try:
                res = minimize(_full_obj_grad, current_params, args=(eps,), method='L-BFGS-B',
                               jac=True, options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.fun < 1e17:
                    current_params = res.x
            except Exception:
                pass
        
        # Final evaluation with eps=0
        if time.time() > _DEADLINE - 5:
            break
        val, _ = _full_obj_grad(current_params, 0.0)
        if val < best_val and np.isfinite(val):
            best_val = val
            best_params = current_params.copy()
            
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
        
    # Convert back to a0, a4, a8
    # We optimized with a0=1.
    # Return [1, u, v]
    return np.array([1.0, best_params[0], best_params[1]])

# EVOLVE-BLOCK-END
