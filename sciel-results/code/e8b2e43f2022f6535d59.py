"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_genlaguerre
import time
from math import comb

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    """
    t_start = time.time()
    deadline = t_start + 590.0  # Safety margin for 600s limit
    
    # The Cohn-Goncalves construction for m=12, n=1 relies on the Laguerre polynomial L_12(x).
    # The optimal double roots are expected to be the zeros of L_12(x).
    
    n = 12
    # Calculate coefficients of L_n(x) = sum_{k=0}^n (-1)^k * binom(n+k, k) / k! * x^k
    # Note: The standard definition is L_n(x) = sum_{k=0}^n (-1)^k * C(n+k, n-k) / k! * x^k
    # which is equivalent to (-1)^k * C(n+k, k) / k! * x^k
    
    coeffs = np.zeros(n + 1)
    for k in range(n + 1):
        # Fix: Handle k=0 case where division by k would occur in previous buggy code
        # Formula: coeff_k = (-1)^k * comb(n+k, k) / factorial(k)
        # To avoid large integers/factorials, we can compute iteratively or use scipy
        # But simple loop is fine for n=12.
        if k == 0:
            coeffs[0] = 1.0
        else:
            # Using the relation: a_k = a_{k-1} * (-(n+k)/(k*(k-1))) ? No, let's stick to the explicit formula
            # a_k = (-1)^k * C(n+k, k) / k!
            # C(n+k, k) = (n+k)! / (n! k!)
            # So a_k = (-1)^k * (n+k)! / (n! (k!)^2)
            # This can be computed recursively:
            # a_0 = 1
            # a_k = a_{k-1} * (-(n+k-1)/k) * (1/k) ? 
            # Let's just use the explicit math.comb and math.factorial for clarity and safety since n is small.
            import math
            coeffs[k] = ((-1)**k) * comb(n + k, k) / math.factorial(k)
            
    # Ensure the highest degree coefficient is non-zero and normalize for root finding stability if needed
    # np.roots expects [a_n, ..., a_0] or [a_0, ..., a_n]? 
    # np.roots expects coefficients in descending order of powers: [a_n, a_{n-1}, ..., a_0]
    # My coeffs array is [a_0, a_1, ..., a_n]
    coeffs_desc = coeffs[::-1]
    
    # Find roots
    try:
        roots_lag = np.roots(coeffs_desc)
        # Filter for real, positive roots
        real_roots = [r.real for r in roots_lag if abs(r.imag) < 1e-8 and r.real > 1e-10]
        real_roots.sort()
        
        if len(real_roots) == 12:
            return real_roots
        else:
            # Fallback to seed if root finding fails or yields wrong count
            return [4.0 * (i + 1) for i in range(12)]
    except Exception:
        # Fallback to seed
        return [4.0 * (i + 1) for i in range(12)]

# EVOLVE-BLOCK-END
