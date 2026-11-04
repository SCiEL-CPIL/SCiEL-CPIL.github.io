import numpy as np
from scipy.special import roots_laguerre

def construct():
    # Compute roots of L_12(x) + L_12'(x)
    # L_n(x) coefficients can be obtained from scipy or manual construction
    # scipy.special.eval_laguerre gives values, not coefficients directly.
    # We can use the recurrence or explicit formula.
    # Or, simpler: use the fact that L_n'(x) = -L_{n-1}^{(1)}(x) (associated Laguerre)
    # So we want roots of L_12(x) - L_{11}^{(1)}(x)
    
    # Let's compute coefficients of L_12(x) and L_11^(1)(x)
    # L_n(x) = sum_{k=0}^n (-1)^k * binom(n, k) * x^k / k!
    # L_n^(alpha)(x) = sum_{k=0}^n (-1)^k * binom(n+alpha, n-k) * x^k / k!
    
    n = 12
    alpha = 1
    
    def laguerre_coeffs(n, alpha=0):
        coeffs = np.zeros(n+1)
        for k in range(n+1):
            # Term: (-1)^k * binom(n+alpha, n-k) * x^k / k!
            # Note: standard definition uses x^k / k!
            # scipy's roots_laguerre uses the standard Laguerre L_n
            # Let's verify the formula.
            # L_n(x) = e^x / n! d^n/dx^n (x^n e^{-x})
            # Coeff of x^k in L_n(x) is (-1)^k * binom(n, k) / k!
            # For L_n^(alpha)(x), coeff is (-1)^k * binom(n+alpha, n-k) / k!
            
            from math import comb, factorial
            val = ((-1)**k) * comb(n+alpha, n-k) / factorial(k)
            coeffs[k] = val
        return coeffs
    
    # L_12(x) coeffs
    c12 = laguerre_coeffs(12, 0)
    # L_11^(1)(x) coeffs
    c11_1 = laguerre_coeffs(11, 1)
    
    # P(x) = L_12(x) - L_11^(1)(x)
    # Degrees: 12 and 11.
    # Pad c11_1 to length 13
    c11_1_padded = np.zeros(13)
    c11_1_padded[:12] = c11_1
    
    # Wait, L_12 has degree 12. L_11^(1) has degree 11.
    # c12 has length 13 (indices 0..12).
    # c11_1 has length 12 (indices 0..11).
    
    # We want roots of L_12(x) - L_11^(1)(x)
    # Note: L_n'(x) = -L_{n-1}^{(1)}(x).
    # So L_12'(x) = -L_11^{(1)}(x).
    # We want roots of L_12(x) + L_12'(x) = L_12(x) - L_11^{(1)}(x).
    
    poly_coeffs = c12 - c11_1_padded
    
    # Find roots
    roots = np.roots(poly_coeffs)
    
    # Filter real positive roots
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 0]
    
    # Should be 12 roots
    if len(real_roots) != 12:
        # Fallback to standard roots if something goes wrong
        std_roots, _ = roots_laguerre(12)
        return [float(r) for r in std_roots]
        
    return [float(r) for r in real_roots]
