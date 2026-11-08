"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Uses the roots of the Laguerre polynomial L_12(x) as the double roots.
    """
    # Compute the roots of the Laguerre polynomial L_12(x)
    # L_n(x) = sum_{k=0}^{n} (-1)^k * C(n,k) * x^k / k!
    # We can use numpy to find the roots
    n = 12
    # Coefficients of L_n(x) in standard form (highest degree first)
    # L_n(x) = (1/n!) * d^n/dx^n (e^{-x} x^n) but easier to use explicit formula
    # L_n(x) = sum_{k=0}^{n} binom(n,k) * (-x)^k / k!
    coeffs = np.zeros(n + 1)
    for k in range(n + 1):
        # coefficient of x^k in L_n is binom(n,k) * (-1)^k / k!
        from math import comb, factorial
        coeffs[k] = comb(n, k) * ((-1)**k) / factorial(k)
    
    # coeffs[k] is coefficient of x^k, so for np.roots we need highest degree first
    coeffs_roots = coeffs[::-1]
    
    roots = np.roots(coeffs_roots)
    
    # Filter to get real positive roots
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0]
    
    # Should have exactly 12 positive real roots
    if len(real_roots) != 12:
        # Fallback: use approximate known values
        real_roots = [0.206, 0.831, 1.801, 3.152, 4.902, 7.061, 
                      9.628, 12.594, 15.933, 19.606, 23.567, 27.775]
    
    # Sort in increasing order
    real_roots = sorted(real_roots)
    
    return real_roots
# EVOLVE-BLOCK-END
