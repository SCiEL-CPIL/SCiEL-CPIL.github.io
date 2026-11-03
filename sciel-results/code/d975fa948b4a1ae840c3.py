"""Optimized program for uncertainty inequality - minimize C4 bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math
from itertools import product

def _hermite_coefficients(n):
    """Return coefficients of physicist's Hermite polynomial H_n(x) as a list [c_0, c_1, ..., c_n]
    such that H_n(x) = sum c_k x^k."""
    # Use recurrence: H_{n+1} = 2x H_n - 2n H_{n-1}
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    
    # H_0 = 1
    h_prev = [1.0]
    # H_1 = 2x
    h_curr = [0.0, 2.0]
    
    for k in range(1, n):
        # H_{k+1} = 2x H_k - 2k H_{k-1}
        # 2x H_k: shift h_curr right by 1, multiply by 2
        term1 = [0.0] * (len(h_curr) + 1)
        for i in range(len(h_curr)):
            term1[i+1] = 2.0 * h_curr[i]
        
        # 2k H_{k-1}: multiply h_prev by 2k
        term2 = [2.0 * k * c for c in h_prev]
        
        # Add term2 to term1 (pad term2 if needed)
        new_len = max(len(term1), len(term2))
        new_h = [0.0] * new_len
        for i in range(len(term1)):
            new_h[i] += term1[i]
        for i in range(len(term2)):
            new_h[i] -= term2[i]
        
        h_prev = h_curr
        h_curr = new_h
    
    return h_curr

def _hermite_val_at_zero(n):
    """Return H_n(0)."""
    coeffs = _hermite_coefficients(n)
    return coeffs[0] if len(coeffs) > 0 else 0.0

def _compute_bound_analytic(coeffs):
    """
    Compute the uncertainty product sqrt(<x^2><k^2>) for g(x) = P(x) exp(-x^2/2).
    coeffs = [a0, a4, a8], a12 is derived from g(0)=0.
    Uses analytic moment calculations for Hermite polynomials.
    """
    a0, a4, a8 = coeffs
    
    # H_n(0) values
    H0_0 = 1.0
    H4_0 = _hermite_val_at_zero(4)   # Should be 12
    H8_0 = _hermite_val_at_zero(8)   # Should be 168
    H12_0 = _hermite_val_at_zero(12) # Should be 10395
    
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    c = [a0, a4, a8, a12]
    degrees = [0, 4, 8, 12]
    
    # Norm squared: ||g||^2 = sum_{i,j} c_i c_j <H_i, H_j>
    # <H_m, H_n> = sqrt(pi) 2^n n! delta_{mn}
    # We can ignore sqrt(pi) as it cancels.
    W = {}
    for d in degrees:
        W[d] = 2.0**d * math.factorial(d)
    
    norm_total = 0.0
    for d in degrees:
        idx = degrees.index(d)
        norm_total += c[idx]**2 * W[d]
    
    if norm_total < 1e-15:
        return 1e10
        
    # <x^2> = (1/norm) sum_{i,j} c_i c_j <H_i, x^2 H_j>
    # <H_m, x^2 H_n> = sqrt(pi) 2^n n! [ (n+1/2) delta_{mn} + (n(n-1)) delta_{m,n-2} + (1/4) delta_{m,n+2} ]
    # Let's verify the indices carefully.
    # x^2 H_n = (1/4) H_{n+2} + (n+1/2) H_n + n(n-1) H_{n-2}
    # So <H_m, x^2 H_n> = sqrt(pi) 2^n n! [ (1/4) delta_{m, n+2} + (n+1/2) delta_{m, n} + n(n-1) delta_{m, n-2} ]
    
    x2_num = 0.0
    for i, di in enumerate(degrees):
        for j, dj in enumerate(degrees):
            term = 0.0
            if di == dj + 2:
                term += 0.25
            if di == dj:
                term += (dj + 0.5)
            if di == dj - 2:
                term += dj * (dj - 1)
            
            if term != 0.0:
                x2_num += c[i] * c[j] * W[dj] * term
    
    x2_mean = x2_num / norm_total
    
    # For even functions, <x> = 0, so var_x = x2_mean
    # Also, for g(x) = P(x) e^{-x^2/2}, the Fourier transform is proportional to P(-i k) e^{-k^2/2} (up to constants).
    # The variance in k-space is the same as in x-space due to the symmetry of the Hermite basis and the Gaussian envelope.
    # Specifically, if g is even, <k> = 0 and <k^2> = <x^2>.
    var_x = x2_mean
    var_k = x2_mean
    
    if var_x < 1e-15 or var_k < 1e-15:
        return 1e10
        
    uncertainty_product = math.sqrt(var_x * var_k)
    return uncertainty_product

def _objective(coeffs):
    try:
        result = _compute_bound_analytic(coeffs)
        if not np.isfinite(result):
            return 1e10
        return result
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin
    
    # Initial guesses
    best_coeffs = np.array([-1.0, 0.0, 0.0])
    best_score = _objective(best_coeffs)
    
    # Local refinement around a few promising points
    candidates = [
        [-1.0, 0.0, 0.0],
        [-1.0, 0.05, 0.0],
        [-1.0, 0.0, 0.01],
        [-0.5, 0.0, 0.0],
        [-1.0, -0.05, 0.0],
        [-1.0, 0.0, -0.01],
    ]
    
    for c in candidates:
        if time.time() > deadline:
            break
        s = _objective(np.array(c))
        if s < best_score:
            best_score = s
            best_coeffs = np.array(c)
    
    # Try differential evolution with tighter bounds and more iterations if time permits
    if time.time() < deadline - 30.0:
        try:
            bounds = [(-5.0, 0.0), (-0.5, 0.5), (-0.5, 0.5)]
            de_result = differential_evolution(
                _objective,
                bounds=bounds,
                maxiter=100,
                tol=1e-12,
                seed=42,
                polish=True,
                workers=1,
                init='sobol',
                updating='immediate'
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_coeffs = de_result.x
        except Exception:
            pass
    
    # Final local polish
    if time.time() < deadline - 5.0:
        try:
            res = minimize(
                _objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
