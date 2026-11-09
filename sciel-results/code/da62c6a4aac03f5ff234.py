"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _I(n):
    """I(n) = ∫_{-∞}^{∞} x^(2n) exp(-x^2) dx = sqrt(pi) * (2n-1)!! / 2^n"""
    if n == 0:
        return np.sqrt(np.pi)
    result = 1.0
    for k in range(1, 2*n, 2):
        result *= k
    return np.sqrt(np.pi) * result / (2.0 ** n)


# Precompute I(0) through I(14) for speed
_I_CACHE = [float(_I(n)) for n in range(15)]


def _compute_raw_value(c0, c4, c8):
    """Compute the raw uncertainty product A(g)*A(F(g)) for the Hermite-Gaussian test function."""
    # P(0) = 0 constraint determines c12
    # H0(0)=1, H4(0)=-12, H8(0)=2520, H12(0)=-103950
    # c0*1 + c4*(-12) + c8*2520 + c12*(-103950) = 0
    c12 = (c0 - 12*c4 + 2520*c8) / 103950.0
    
    # Build P(y) where y=x^2, P(x) = sum p_i * x^(2i)
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 420x^4 - 1680x^2 + 105
    # H12(x) = x^12 - 66x^10 + 3960x^8 - 118800x^6 + 1188000x^4 - 3991680x^2 + 103950
    
    p = np.zeros(7)
    # Constant term (x^0): c0*1 + c4*3 + c8*105 + c12*103950
    p[0] = c0 + 3*c4 + 105*c8 + 103950*c12
    # x^2 term: c4*(-6) + c8*(-1680) + c12*(-3991680)
    p[1] = -6*c4 - 1680*c8 - 3991680*c12
    # x^4 term: c4*1 + c8*420 + c12*1188000
    p[2] = c4 + 420*c8 + 1188000*c12
    # x^6 term: c8*(-28) + c12*(-118800)
    p[3] = -28*c8 - 118800*c12
    # x^8 term: c8*1 + c12*3960
    p[4] = c8 + 3960*c12
    # x^10 term: c12*(-66)
    p[5] = -66*c12
    # x^12 term: c12*1
    p[6] = c12
    
    # P(x)^2 = (sum_i p_i x^(2i))^2 = sum_{i,j} p_i p_j x^(2(i+j))
    q = np.zeros(13)
    for i in range(7):
        for j in range(7):
            if i + j <= 12:
                q[i+j] += p[i] * p[j]
    
    # norm_sq = ∫|g|^2 dx = sum_m q_m I(m)
    norm_sq = 0.0
    for m in range(13):
        norm_sq += q[m] * _I_CACHE[m]
    
    # x2_norm = ∫x^2|g|^2 dx = sum_m q_m I(m+1)
    x2_norm = 0.0
    for m in range(13):
        x2_norm += q[m] * _I_CACHE[m+1]
    
    if norm_sq <= 0:
        return 1e10
    
    # A(g)^2 = x2_norm/norm_sq (since <x>=0)
    # For even functions with this structure, A(F(g))^2 = x2_norm/norm_sq
    # So A(g)*A(F(g)) = x2_norm/norm_sq
    return x2_norm / norm_sq


def construct():
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin before 210s limit
    
    # Seed values
    seed = np.array([-113.0/100, 1.0/25, 1.0/3240])
    
    # Verify seed gives ~1.0
    seed_val = _compute_raw_value(*seed)
    
    # If our formula doesn't give ~1.0 for the seed, something is wrong
    # In that case, just return the seed
    if abs(seed_val - 1.0) > 0.1:
        return seed
    
    # Local search to minimize the raw value
    best_x = seed.copy()
    best_val = seed_val
    
    def objective(x):
        return _compute_raw_value(x[0], x[1], x[2])
    
    # Phase 1: Differential Evolution for global exploration
    if time.time() < deadline - 120:
        try:
            bounds = [(-10, 10), (-10, 10), (-10, 10)]
            de_result = differential_evolution(objective, bounds, seed=42, 
                                               maxiter=100, tol=1e-10,
                                               polish=True, init='sobol')
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_x = de_result.x
        except Exception:
            pass
    
    if time.time() > deadline - 60:
        return best_x
    
    # Phase 2: Nelder-Mead for local refinement
    try:
        result = minimize(objective, best_x, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    except Exception:
        pass
    
    if time.time() > deadline - 30:
        return best_x
    
    # Phase 3: BFGS polish
    try:
        result = minimize(objective, best_x, method='BFGS',
                         options={'maxiter': 5000, 'gtol': 1e-14})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        return best_x
    
    # Phase 4: L-BFGS-B with very tight tolerance for final polish
    try:
        result = minimize(objective, best_x, method='L-BFGS-B',
                         options={'maxiter': 10000, 'gtol': 1e-16, 'ftol': 1e-16})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    except Exception:
        pass
    
    return best_x
# EVOLVE-BLOCK-END
