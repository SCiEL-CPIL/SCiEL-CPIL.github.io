# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.special import eval_hermite

T_START = time.time()
DEADLINE = 210.0 - 15.0

def _now():
    return time.time()

def _time_left():
    return DEADLINE - _now()

def _H(n, x):
    return eval_hermite(n, x)

def _coeff12(c0, c4, c8):
    h0 = _H(0, 0.0)
    h4 = _H(4, 0.0)
    h8 = _H(8, 0.0)
    h12 = _H(12, 0.0)
    if abs(h12) < 1e-15:
        return 0.0
    return -(c0 * h0 + c4 * h4 + c8 * h8) / h12

def _uncertainty_product_analytic(c0, c4, c8):
    """
    Analytic calculation of the uncertainty product using the orthogonality
    and recurrence properties of Hermite-Gaussian functions.
    Var_x = <x^2 f, f> / <f, f>
    Product = Var_x * Var_p = Var_x^2 (for FT-invariant magnitude)
    
    For f = sum c_n H_n(x) exp(-x^2/2), the inner products involve factors of 2^n n! sqrt(pi).
    Since these cancel in the ratio, we compute:
    Var_x = sum(c_n^2 * 2^n * n! * (n+0.5)) / sum(c_n^2 * 2^n * n!)
    """
    c12 = _coeff12(c0, c4, c8)
    
    # Coefficients for the full combination
    cs = [c0, c4, c8, c12]
    indices = [0, 4, 8, 12]
    
    # <f, f> = sum c_i^2 * 2^i * i! (the sqrt(pi) factor cancels in the ratio)
    norm_sq = 0.0
    for c, n in zip(cs, indices):
        norm_sq += c * c * (2.0 ** n) * math.factorial(n)
    
    if norm_sq < 1e-15:
        return float('inf')
        
    # <x^2 f, f>
    # In the orthonormal basis, <phi_m | x^2 | phi_n> has diagonal entry (n+0.5)
    # and off-diagonal entries connecting n to n±2.
    # Since our indices {0,4,8,12} differ by 4, there are no off-diagonal connections.
    # So <x^2 f, f> = sum c_n^2 * 2^n * n! * (n+0.5)
    x2_moment = 0.0
    for c, n in zip(cs, indices):
        x2_moment += (n + 0.5) * c * c * (2.0 ** n) * math.factorial(n)
        
    var_x = x2_moment / norm_sq
    return var_x * var_x

def _score(c0, c4, c8):
    try:
        return _uncertainty_product_analytic(c0, c4, c8)
    except Exception:
        return float('inf')

def construct():
    # The analytic minimum is expected at c4=0, c8=0
    # Let's verify with a few points
    candidates = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (-113.0/100.0, 1.0/25.0, 1.0/3240.0), # Seed
        (1.0, 0.1, 0.0),
        (1.0, 0.0, 0.1),
    ]
    
    best = candidates[0]
    best_val = _score(*best)
    
    for c in candidates[1:]:
        v = _score(*c)
        if v < best_val:
            best_val = v
            best = c
            
    # Quick Nelder-Mead on 3D space just in case
    from scipy.optimize import minimize
    def obj(p):
        return _score(p[0], p[1], p[2])
        
    res = minimize(obj, best, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
    if res.fun < best_val:
        best = (res.x[0], res.x[1], res.x[2])
        best_val = res.fun
        
    return np.array(best)
# EVOLVE-BLOCK-END
