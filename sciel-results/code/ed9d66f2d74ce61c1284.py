"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 195.0  # Safety margin before 210s kill

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
# We need n up to 12 (since max degree in P^2 is 12+12=24 -> x^{24}, n=12)
_I = np.zeros(13)
for _n in range(13):
    log_val = 0.0
    for _m in range(1, 2*_n + 1):
        log_val += np.log(_m)
    log_val -= _n * np.log(4.0)
    for _m in range(1, _n + 1):
        log_val -= np.log(_m)
    log_val += 0.5 * np.log(np.pi)
    _I[_n] = np.exp(log_val)

# Precompute the coefficient matrix for p_k in terms of (a0, a4, a8, a12)
# p0 = a0 + 3*a4 + 105*a8 + 945*a12
# p1 = -6*a4 - 420*a8 - 7560*a12
# p2 = a4 + 210*a8 + 13860*a12
# p3 = -28*a8 - 6930*a12
# p4 = a8 + 1155*a12
# p5 = -66*a12
# p6 = a12
# 
# With a12 = -(a0 + 3*a4 + 105*a8)/945, we can express p in terms of a0,a4,a8
# Let's compute the 7x3 matrix M such that p = M @ [a0, a4, a8]
_M = np.zeros((7, 3))
# For each k, p_k = c_k0*a0 + c_k1*a4 + c_k2*a8
# Substitute a12 = -(a0 + 3*a4 + 105*a8)/945

# p0 = a0 + 3*a4 + 105*a8 + 945*(-(a0+3*a4+105*a8)/945) = 0
# So p0 is identically zero (g(0)=0 constraint)
_M[0] = [0.0, 0.0, 0.0]

# p1 = -6*a4 - 420*a8 - 7560*(-(a0+3*a4+105*a8)/945)
#    = -6*a4 - 420*a8 + 8*(a0+3*a4+105*a8)
#    = 8*a0 + (-6+24)*a4 + (-420+840)*a8
#    = 8*a0 + 18*a4 + 420*a8
_M[1] = [8.0, 18.0, 420.0]

# p2 = a4 + 210*a8 + 13860*(-(a0+3*a4+105*a8)/945)
#    = a4 + 210*a8 - 14.666...*(a0+3*a4+105*a8)
#    13860/945 = 14.666... = 44/3
#    = -44/3*a0 + (1 - 44)*a4 + (210 - 44*105)*a8
#    = -44/3*a0 + (-43)*a4 + (210 - 4620)*a8
#    = -44/3*a0 - 43*a4 - 4410*a8
_M[2] = [-44.0/3.0, -43.0, -4410.0]

# p3 = -28*a8 - 6930*(-(a0+3*a4+105*a8)/945)
#    = -28*a8 + 7.333...*(a0+3*a4+105*a8)
#    6930/945 = 7.333... = 22/3
#    = 22/3*a0 + 22*a4 + (-28 + 22*105)*a8
#    = 22/3*a0 + 22*a4 + (-28 + 2310)*a8
#    = 22/3*a0 + 22*a4 + 2282*a8
_M[3] = [22.0/3.0, 22.0, 2282.0]

# p4 = a8 + 1155*(-(a0+3*a4+105*a8)/945)
#    = a8 - 1.2222...*(a0+3*a4+105*a8)
#    1155/945 = 1.2222... = 11/9
#    = -11/9*a0 - 11/3*a4 + (1 - 11)*a8
#    = -11/9*a0 - 11/3*a4 - 10*a8
_M[4] = [-11.0/9.0, -11.0/3.0, -10.0]

# p5 = -66*(-(a0+3*a4+105*a8)/945)
#    = 66/945*(a0+3*a4+105*a8)
#    66/945 = 2/28.18... = 22/315
#    = 22/315*a0 + 66/315*a4 + 2310/315*a8
#    = 22/315*a0 + 22/105*a4 + 22/3*a8
_M[5] = [22.0/315.0, 22.0/105.0, 22.0/3.0]

# p6 = -(a0+3*a4+105*a8)/945
#    = -1/945*a0 - 3/945*a4 - 105/945*a8
#    = -1/945*a0 - 1/315*a4 - 1/9*a8
_M[6] = [-1.0/945.0, -1.0/315.0, -1.0/9.0]

# q_k = p_k * (-1)^k
_q_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
_Mq = _M * _q_signs[:, np.newaxis]


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945 (enforces g(0)=0)
    
    A(f)^2 = <x^2>/<1> for |g|^2
    A(F)^2 = <xi^2>/<1> for |G|^2
    """
    a0, a4, a8 = params[0], params[1], params[2]
    
    # Compute p coefficients
    p = _M @ params
    q = _Mq @ params
    
    # m_k = sum_{i,j=0}^{6} p_i p_j I[k+i+j]
    # n_k = 2*pi * sum_{i,j=0}^{6} q_i q_j I[k+i+j]
    
    m = np.zeros(3)
    n = np.zeros(3)
    
    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0.0 and qi == 0.0:
            continue
        for j in range(7):
            idx0 = i + j
            if idx0 >= 13:
                continue
            m[0] += pi * p[j] * _I[idx0]
            n[0] += 2.0 * np.pi * qi * q[j] * _I[idx0]
            idx1 = i + j + 1
            if idx1 < 13:
                m[1] += pi * p[j] * _I[idx1]
                n[1] += 2.0 * np.pi * qi * q[j] * _I[idx1]
            idx2 = i + j + 2
            if idx2 < 13:
                m[2] += pi * p[j] * _I[idx2]
                n[2] += 2.0 * np.pi * qi * q[j] * _I[idx2]
    
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
    # d(bound)/d(a_c) = (1/(2*bound)) * (vF * d(vf)/d(a_c) + vf * d(vF)/d(a_c))
    # d(vf)/d(a_c) = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # d(vF)/d(a_c) = (dn2/da_c * n0 - n2 * dn0/da_c) / n0^2
    
    grad = np.zeros(3)
    
    for c in range(3):
        # dp_k/da_c = M[k, c]
        # dq_k/da_c = Mq[k, c]
        dp = _M[:, c]
        dq = _Mq[:, c]
        
        # dm_k/da_c = 2*sum_i dp_i * sum_j p_j I[k+i+j]
        dm = np.zeros(3)
        for i in range(7):
            dpi = dp[i]
            if dpi == 0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 13:
                        s += p[j] * _I[idx]
                dm[k] += 2.0 * dpi * s
        
        dn = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if dqi == 0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 13:
                        s += q[j] * _I[idx]
                dn[k] += 2.0 * 2.0 * np.pi * dqi * s
        
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
    # Use Differential Evolution for global search
    # Bounds: The coefficients have different scales.
    # a0 ~ 1, a4 ~ 0.04, a8 ~ 0.0003.
    # We use wide bounds to ensure coverage.
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    best_val = np.inf
    best_params = None
    
    # 1. Global Search with Differential Evolution
    if time.time() < _DEADLINE - 60:
        try:
            de_res = differential_evolution(
                _objective, 
                bounds, 
                seed=42, 
                maxiter=1000, 
                popsize=15, 
                tol=1e-12,
                polish=False, # We will do our own polish
                init='sobol' # Better distribution
            )
            if de_res.fun < best_val and np.isfinite(de_res.fun):
                best_val = de_res.fun
                best_params = de_res.x.copy()
        except Exception:
            pass
    
    # 2. Local Polish with L-BFGS-B from best DE result
    if best_params is not None and time.time() < _DEADLINE - 20:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    # 3. Fallback to seed if DE failed or found nothing
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
        
    return np.array(best_params)
# EVOLVE-BLOCK-END
