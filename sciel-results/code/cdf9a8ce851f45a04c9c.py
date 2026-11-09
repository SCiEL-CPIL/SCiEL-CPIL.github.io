"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    """
    Returns 12 double-root locations z_1..z_12.
    Strategy: Use the 12 smallest positive zeros of the generalized Laguerre polynomial
    L_12^{(-1/2)}(x). These are the optimal Gaussian quadrature nodes for the weight
    x^{-1/2} e^{-x}, which should minimize the residual quotient q(x) and thus B.
    """
    
    # We need the roots of L_12^{(-1/2)}(x).
    # The generalized Laguerre polynomial L_n^{(alpha)}(x) can be computed.
    # For alpha = -1/2, n = 12.
    # We can use numpy's polynomial roots or scipy.special.roots_genlaguerre.
    
    try:
        from scipy.special import roots_genlaguerre
        # roots_genlaguerre(n, alpha) returns the roots of L_n^{(alpha)}(x)
        # n=12, alpha=-0.5
        x, w = roots_genlaguerre(12, -0.5)
        z = [float(xi) for xi in x]
    except ImportError:
        # Fallback: compute L_12^{(-1/2)}(x) coefficients and find roots using numpy
        import numpy as np
        
        # Coefficients of L_n^{(alpha)}(x)
        # L_n^{(alpha)}(x) = sum_{k=0}^n (-1)^k * binom(n+alpha, n-k) * x^k / k!
        # For n=12, alpha=-0.5
        n = 12
        alpha = -0.5
        coeffs = []
        for k in range(n, -1, -1):  # descending powers for numpy.poly1d
            # term = (-1)^k * C(n+alpha, n-k) * x^k / k!
            # C(n+alpha, n-k) = Gamma(n+alpha+1) / (Gamma(n-k+1) * Gamma(k+alpha+1))
            # Using math.lgamma for log-gamma
            log_num = math.lgamma(n + alpha + 1)
            log_den1 = math.lgamma(n - k + 1)
            log_den2 = math.lgamma(k + alpha + 1)
            log_binom = log_num - log_den1 - log_den2
            binom = math.exp(log_binom)
            
            log_fact = math.lgamma(k + 1)
            fact = math.exp(log_fact)
            
            term_coeff = ((-1) ** k) * binom / fact
            coeffs.append(term_coeff)
        
        # coeffs are for x^n ... x^0
        # Find roots
        roots = np.roots(coeffs)
        # Filter for real, positive roots
        z = []
        for r in roots:
            if abs(r.imag) < 1e-10 and r.real > 0:
                z.append(float(r.real))
        z.sort()
    
    # Ensure we have exactly 12 distinct positive finite values
    if len(z) != 12:
        # Fallback to Bessel zeros if something went wrong
        z = [float((i * math.pi) ** 2) for i in range(1, 13)]
    
    # Verify conditions
    for zi in z:
        if not math.isfinite(zi) or zi <= 0:
            # Fallback
            z = [float((i * math.pi) ** 2) for i in range(1, 13)]
            break
    
    # Check pairwise distinctness
    z_sorted = sorted(z)
    for i in range(len(z_sorted) - 1):
        if abs(z_sorted[i] - z_sorted[i+1]) < 1e-12:
            z = [float((i * math.pi) ** 2) for i in range(1, 13)]
            break
    
    return z
# EVOLVE-BLOCK-END
