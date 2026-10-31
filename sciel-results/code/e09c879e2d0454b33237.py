import numpy as np
from scipy.optimize import minimize
import time

def _compute_objective(coeffs):
    """
    Compute the uncertainty product for the given Hermite coefficients.
    coeffs: array of [a0, a4, a8]. a12 is derived to satisfy g(0)=0.
    """
    a0, a4, a8 = coeffs
    # Calculate a12 such that g(0) = 0
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=945
    # g(0) = a0*1 + a4*3 + a8*105 + a12*945 = 0
    a12 = -(a0 + 3*a4 + 105*a8) / 945.0
    
    # Coefficients for h_n where n is in {0, 4, 8, 12}
    # c_n is the coefficient of h_n(x) = H_n(x) exp(-x^2/2)
    c = np.zeros(13)
    c[0] = a0
    c[4] = a4
    c[8] = a8
    c[12] = a12
    
    # Precompute sqrt(pi) * n! for n=0..12
    # log_fact(n) = log(n!)
    # We need sum c_k^2 * sqrt(pi) * k!
    # Let's compute norms directly
    sqrt_pi = np.sqrt(np.pi)
    fact = np.array([1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0, 3628800.0, 39916800.0, 479001600.0])
    norms_sq = sqrt_pi * fact
    
    # 1. ||g||_2^2
    norm_g_sq = np.sum(c**2 * norms_sq)
    if norm_g_sq < 1e-10:
        return 1e10 # Avoid division by zero
    
    # 2. ||g'||_2^2
    # d_k = (k+1) c_{k+1} - 0.5 c_{k-1}
    d = np.zeros(13)
    for k in range(13):
        term1 = 0.0
        term2 = 0.0
        if k + 1 < 13:
            term1 = (k + 1) * c[k + 1]
        if k - 1 >= 0:
            term2 = -0.5 * c[k - 1]
        d[k] = term1 + term2
    norm_gp_sq = np.sum(d**2 * norms_sq)
    
    # 3. ||x g||_2^2
    # e_k = 0.5 c_{k-1} + (k+1) c_{k+1}
    e = np.zeros(13)
    for k in range(13):
        term1 = 0.0
        term2 = 0.0
        if k - 1 >= 0:
            term1 = 0.5 * c[k - 1]
        if k + 1 < 13:
            term2 = (k + 1) * c[k + 1]
        e[k] = term1 + term2
    norm_xg_sq = np.sum(e**2 * norms_sq)
    
    # Objective: Product of uncertainties
    # Delta_x = sqrt(||xg||^2 / ||g||^2)
    # Delta_p = sqrt(||g'||^2 / ||g||^2)
    # Product = sqrt(||xg||^2 ||g'||^2) / ||g||
    # Minimizing the product is equivalent to minimizing the square of the product
    obj = (norm_xg_sq * norm_gp_sq) / norm_g_sq
    
    return obj

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 180.0 # 180s limit, keep margin
    
    # Seed values
    seed_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_score = _compute_objective(seed_coeffs)
    best_coeffs = seed_coeffs.copy()
    
    # Define bounds to prevent coefficients from exploding
    # The seed values are roughly [-1.13, 0.04, 0.0003]
    # We allow a reasonable range
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    # Multiple restarts to find global minimum
    # Initial guesses: seed, zeros, random
    initial_guesses = [
        seed_coeffs,
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([1.0, -0.1, -0.001]),
    ]
    
    # Add some random restarts
    np.random.seed(42)
    for _ in range(10):
        r = np.random.uniform(-2, 2, 3)
        # Scale smaller for higher degrees to match typical magnitudes
        r[1] *= 0.1
        r[2] *= 0.001
        initial_guesses.append(r)
        
    # Optimization loop
    for x0 in initial_guesses:
        if time.time() > deadline - 30: # 30s margin
            break
            
        try:
            res = minimize(_compute_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
            
        # Try L-BFGS-B if Nelder-Mead didn't converge well or for polish
        if time.time() > deadline - 20:
            break
        try:
            res2 = minimize(_compute_objective, best_coeffs, method='L-BFGS-B', 
                            bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12})
            if res2.success and res2.fun < best_score:
                best_score = res2.fun
                best_coeffs = res2.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
