"""Optimized uncertainty bound using Differential Evolution and L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 200

def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty bound for given coefficients.
    Returns the bound value.
    """
    # a12 is derived from P(0)=0
    # Using the coefficients from the original code's expansion
    # p[0] = a0 + 3*a4 + 105*a8 + 945*a12 = 0 => a12 = -(a0 + 3*a4 + 105*a8) / 945
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Polynomial coefficients p[k] for x^(2k)
    # Based on the original code's expansion which seemed to follow a specific Hermite basis
    # We stick to the original code's expansion to ensure consistency with the task's expected math,
    # but we verify the logic.
    p = np.zeros(7)
    p[0] = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    
    # Calculate moments
    # m_k = integral x^(2k) P(x)^2 e^(-x^2) dx
    # n_k = 2*pi * integral x^(2k) Q(x)^2 e^(-x^2) dx
    # Q(x) = P(-ix) => q_k = p_k * (-1)^k
    
    sp = np.sqrt(np.pi)
    I = np.zeros(13)
    I[0] = sp
    for n in range(1, 13):
        I[n] = I[n-1] * (2*n - 1) / (2*n)
        
    m0 = 0.0
    m2 = 0.0
    n0 = 0.0
    n2 = 0.0
    
    q = np.array([p[k] * ((-1.0)**k) for k in range(7)])
    
    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0 and qi == 0:
            continue
        for j in range(7):
            pji = p[j]
            qji = q[j]
            if pji == 0 and qji == 0:
                continue
            
            # m0: k=0
            idx0 = i + j
            if idx0 <= 12:
                m0 += pi * pji * I[idx0]
                n0 += 2.0 * np.pi * qi * qji * I[idx0]
                
            # m2: k=1 => x^2 * x^(2i) * x^(2j) = x^(2(i+j+1)) => idx = 1 + i + j
            idx1 = 1 + i + j
            if idx1 <= 12:
                m2 += pi * pji * I[idx1]
                n2 += 2.0 * np.pi * qi * qji * I[idx1]
                
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
        
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18
        
    return np.sqrt(vf * vF)

def _objective(params):
    a0, a4, a8 = params
    return _compute_bound(a0, a4, a8)

def _objective_grad(params):
    a0, a4, a8 = params
    val = _objective(params)
    h = 1e-8
    grad = np.zeros(3)
    for i in range(3):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += h
        p_minus[i] -= h
        val_plus = _objective(p_plus)
        val_minus = _objective(p_minus)
        grad[i] = (val_plus - val_minus) / (2*h)
    return val, grad

def construct():
    # Strategy:
    # 1. Differential Evolution on (a4, a8) with a0 fixed to 1.0
    # 2. L-BFGS-B polish on (a0, a4, a8)
    
    # Bounds for DE
    # a4 and a8 are likely small.
    bounds = [(-1.0, 1.0), (-0.5, 0.5)]
    
    def de_objective(x):
        a4, a8 = x
        return _compute_bound(1.0, a4, a8)
        
    try:
        # Run DE
        result_de = differential_evolution(de_objective, bounds, maxiter=100, popsize=15, seed=42, polish=False)
        best_a4, best_a8 = result_de.x
        best_val = result_de.fun
    except Exception:
        # Fallback to seed-like values
        best_a4, best_a8 = 0.04, 0.0003
        best_val = _compute_bound(1.0, best_a4, best_a8)
        
    # Initial guess for L-BFGS
    start_params = np.array([1.0, best_a4, best_a8])
    
    # L-BFGS-B Polish
    try:
        if time.time() < _DEADLINE - 20:
            res = minimize(_objective_grad, start_params, method='L-BFGS-B', jac=True, 
                          options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
            else:
                best_params = start_params
        else:
            best_params = start_params
    except Exception:
        best_params = start_params
        
    # Final Nelder-Mead polish
    try:
        if time.time() < _DEADLINE - 5:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
    except Exception:
        pass
        
    if not np.all(np.isfinite(best_params)):
        return np.array([1.0, 0.04, 0.0003])
        
    return np.array(best_params)
# EVOLVE-BLOCK-END
