import numpy as np
import time
from scipy.optimize import differential_evolution, minimize, dual_annealing

_START_TIME = time.time()
_DEADLINE = _START_TIME + 200  # Safety margin: evaluator kills at 210s

def _time_left():
    return _DEADLINE - time.time()

# Hermite polynomial coefficients for H0, H4, H8, H12
H0 = np.array([1.0])
H4 = np.array([12.0, 0.0, -48.0, 0.0, 16.0])
H8 = np.array([10080.0, 0.0, -34560.0, 0.0, 23040.0, 0.0, -4608.0, 0.0, 256.0])
H12 = np.array([10321920.0, 0.0, -34406400.0, 0.0, 34406400.0, 0.0, -12902400.0, 0.0, 2090112.0, 0.0, -147456.0, 0.0, 4096.0])

H0_0 = 1.0
H4_0 = 12.0
H8_0 = 3024.0
H12_0 = 10321920.0

def _poly_mul(p, q):
    if len(p) == 0 or len(q) == 0:
        return np.array([0.0])
    return np.convolve(p, q)

def _poly_deriv(p):
    if len(p) <= 1:
        return np.array([0.0])
    return np.arange(1, len(p)) * p[1:]

def _poly_mul_by_x(p):
    return np.concatenate(([0.0], p))

def _integrate_poly_gaussian(p):
    """
    Compute integral of P(x) * exp(-x^2) dx from -inf to inf.
    Only even powers contribute.
    """
    result = 0.0
    sqrt_pi = np.sqrt(np.pi)
    for k, c in enumerate(p):
        if c == 0.0:
            continue
        if k % 2 != 0:
            continue
        m = k // 2
        if m == 0:
            i_k = sqrt_pi
        else:
            double_fact = 1.0
            for i in range(1, 2*m, 2):
                double_fact *= i
            i_k = double_fact * sqrt_pi / (2.0 ** m)
        result += c * i_k
    return result

def _precompute_gram_matrices():
    """
    Precompute Gram matrices G, A, M for the basis {H0, H4, H8, H12}
    with respect to weight exp(-x^2).
    
    G[i][j] = <h_i, h_j>
    A[i][j] = <x^2 h_i, h_j>
    M[i][j] = <(h_i' - x h_i), (h_m' - x h_m)>
    """
    basis = [H0, H4, H8, H12]
    n = 4
    
    G = np.zeros((n, n))
    A = np.zeros((n, n))
    M = np.zeros((n, n))
    
    for i in range(n):
        hi = basis[i]
        hi_deriv = _poly_deriv(hi)
        x_hi = _poly_mul_by_x(hi)
        # M operator: (h_i' - x h_i)
        max_len = max(len(hi_deriv), len(x_hi))
        Qi = np.zeros(max_len)
        Qi[:len(hi_deriv)] += hi_deriv
        Qi[:len(x_hi)] -= x_hi
        
        x2_hi = _poly_mul_by_x(_poly_mul_by_x(hi))
        
        for j in range(n):
            hj = basis[j]
            hj_deriv = _poly_deriv(hj)
            x_hj = _poly_mul_by_x(hj)
            max_len_j = max(len(hj_deriv), len(x_hj))
            Qj = np.zeros(max_len_j)
            Qj[:len(hj_deriv)] += hj_deriv
            Qj[:len(x_hj)] -= x_hj
            
            # G[i][j] = <hi, hj>
            prod = _poly_mul(hi, hj)
            G[i][j] = _integrate_poly_gaussian(prod)
            
            # A[i][j] = <x^2 hi, hj>
            prod_x2 = _poly_mul(x2_hi, hj)
            A[i][j] = _integrate_poly_gaussian(prod_x2)
            
            # M[i][j] = <Qi, Qj>
            prod_M = _poly_mul(Qi, Qj)
            M[i][j] = _integrate_poly_gaussian(prod_M)
    
    return G, A, M

G, A, M = _precompute_gram_matrices()

def _uncertainty_product_fast(a0, a4, a8):
    """
    Compute sigma_x * sigma_p for g(x) = P(x) exp(-x^2/2)
    using precomputed Gram matrices.
    """
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    c = np.array([a0, a4, a8, a12])
    
    # Check if c is zero vector
    if np.allclose(c, 0.0, atol=1e-15):
        return 1e10
    
    cGc = c @ G @ c
    if cGc <= 1e-10:
        return 1e10
    
    cAc = c @ A @ c
    cMc = c @ M @ c
    
    sigma_x_sq = cAc / cGc
    sigma_p_sq = cMc / cGc
    
    if sigma_x_sq < 0 or sigma_p_sq < 0:
        return 1e10
    
    return np.sqrt(sigma_x_sq * sigma_p_sq)

def construct():
    seed_result = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_result = seed_result.copy()
    best_score = _uncertainty_product_fast(*seed_result)
    
    # 1. Differential Evolution - Global Search
    if _time_left() > 30:
        try:
            # Search in a reasonable range around the seed
            # Scale by norm of seed
            seed_norm = np.linalg.norm(seed_result)
            scale = seed_norm * 5  # Explore within 5x seed scale
            
            bounds = [(-scale, scale), (-scale, scale), (-scale, scale)]
            
            de_result = differential_evolution(
                _uncertainty_product_fast,
                bounds,
                seed=42,
                maxiter=200,
                tol=1e-8,
                polish=True,
                init='sobol',
                popsize=30,
                mutation=(0.5, 1.0),
                recombination=0.8
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_result = de_result.x.copy()
        except Exception:
            pass
    
    # 2. Dual Annealing - Another global method
    if _time_left() > 20:
        try:
            seed_norm = np.linalg.norm(seed_result)
            scale = seed_norm * 5
            
            bounds_da = [(-scale, scale), (-scale, scale), (-scale, scale)]
            
            da_result = dual_annealing(
                _uncertainty_product_fast,
                bounds=bounds_da,
                seed=42,
                maxiter=500,
                restarts=5,
                no_local_search=False
            )
            if da_result.fun < best_score:
                best_score = da_result.fun
                best_result = da_result.x.copy()
        except Exception:
            pass
    
    # 3. Local refinement from best found using Powell
    if _time_left() > 10:
        try:
            res = minimize(
                _uncertainty_product_fast,
                best_result,
                method='Powell',
                options={'maxiter': 10000, 'xtol': 1e-12, 'ftol': 1e-14}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # 4. Final high-precision polish with Nelder-Mead
    if _time_left() > 5:
        try:
            res = minimize(
                _uncertainty_product_fast,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    return best_result
