# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195  # Safety margin before 210s kill

def _precompute_I():
    """Precompute I[n] = integral x^{2n} e^{-x^2} dx for n=0..13."""
    sp = math.sqrt(math.pi)
    I = np.zeros(14)
    for n in range(14):
        if n == 0:
            I[n] = sp
        else:
            log_val = 0.0
            for m in range(1, 2 * n + 1):
                log_val += math.log(m)
            log_val -= n * math.log(4.0)
            for m in range(1, n + 1):
                log_val -= math.log(m)
            log_val += 0.5 * math.log(math.pi)
            I[n] = math.exp(log_val)
    return I

_I = _precompute_I()

def _bound_and_grad(params):
    a0, a4, a8 = params[0], params[1], params[2]
    
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    p = np.zeros(7)
    p[0] = 0.0 # Enforced
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    p[3] = -28.0 * a8 - 6930.0 * a12
    p[4] = a8 + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = a12
    
    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        if pi == 0.0:
            continue
        for j in range(7):
            pij = pi * p[j]
            if pij == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    m[k] += pij * _I[idx]
    
    q = p * np.array([1, -1, 1, -1, 1, -1, 1])
    
    n = np.zeros(3)
    factor = 2.0 * math.pi
    for i in range(7):
        qi = q[i]
        if qi == 0.0:
            continue
        for j in range(7):
            qij = qi * q[j]
            if qij == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    n[k] += factor * qij * _I[idx]
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 1e-15 or n0 <= 1e-15 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = math.sqrt(vf * vF)
    
    da12_da = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    H_coeffs = np.array([
        [1, 0, 0, 0, 0, 0, 0],      
        [3, -6, 1, 0, 0, 0, 0],     
        [105, -420, 210, -28, 1, 0, 0], 
        [945, -7560, 13860, -6930, 1155, -66, 1] 
    ])
    
    grad = np.zeros(3)
    
    for c in range(3):
        dp = H_coeffs[c] + H_coeffs[3] * da12_da[c]
        
        dm = np.zeros(3)
        for i in range(7):
            dpi = dp[i]
            if dpi == 0.0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        s += p[j] * _I[idx]
                dm[k] += 2.0 * dpi * s
        
        dq = dp * np.array([1, -1, 1, -1, 1, -1, 1])
        
        dn = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if dqi == 0.0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        s += q[j] * _I[idx]
                dn[k] += 2.0 * factor * dqi * s
        
        dvm2 = (dm[1] * m0 - m2 * dm[0]) / (m0 * m0)
        dvn2 = (dn[1] * n0 - n2 * dn[0]) / (n0 * n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad

def _objective(params):
    val, _ = _bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 175  # Strict internal deadline
    
    # Bounds for differential evolution
    # Based on the structure, coefficients shouldn't be huge
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: Global Search with Differential Evolution
    if time.time() < deadline - 60:
        try:
            de_res = differential_evolution(_objective, bounds, maxiter=50, popsize=20, tol=1e-8, seed=42, polish=False)
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_params = de_res.x.copy()
        except Exception as e:
            pass

    # Phase 2: Local Refinement from best DE result and some heuristic starts
    starts = []
    if best_params is not None:
        starts.append(best_params)
    
    # Add heuristic starts
    heuristic_starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.05, 0.001]),
        np.array([-2.0, 0.1, 0.002]),
    ]
    starts.extend(heuristic_starts)
    
    for start in starts:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
        
    # Phase 3: High precision polish
    if time.time() < deadline - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    return best_params

# EVOLVE-BLOCK-END
