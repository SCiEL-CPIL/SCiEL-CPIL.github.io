"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Return the 12 positive double-root locations for the Laguerre combination
    (m=12, n=1). These are the zeros of the generalized Laguerre polynomial
    L_12^(0)(x), which provide the physically consistent optimal baseline
    for the Cohn-Goncalves Fourier sign uncertainty construction.
    """
    # Compute the 12 zeros of L_12^(0)(x)
    # numpy.polynomial.laguerre.laggauss gives nodes (zeros) of L_n^(1)(x)
    # We need zeros of L_12^(0)(x). Use the companion matrix or root-finding.
    # 
    # For L_n^(alpha)(x), we can use scipy.special.roots_genlaguerre
    # or compute via numpy's polynomial utilities.
    
    # Using scipy for robust root finding of generalized Laguerre polynomials
    try:
        from scipy.special import roots_genlaguerre
        # roots_genlaguerre(n, alpha) returns roots of L_n^(alpha)(x)
        roots = roots_genlaguerre(12, 0.0)
        # The roots are returned as real values, all positive for alpha > -1
        roots = np.sort(roots.real)
        return [float(r) for r in roots]
    except ImportError:
        # Fallback: compute roots of L_12^(0)(x) using numpy
        # L_n^(0)(x) = sum_{k=0}^{n} (-1)^k * binom(n,k) / k! * x^k
        # Build the polynomial coefficients
        n = 12
        coeffs = np.zeros(n + 1)
        for k in range(n + 1):
            coeffs[n - k] = ((-1) ** k) * (np.math.factorial(n) / (np.math.factorial(n - k) * np.math.factorial(k))) / np.math.factorial(k)
        # Find roots
        r = np.roots(coeffs)
        # Filter to positive real roots
        r = r[np.abs(r.imag) < 1e-10]
        r = r.real
        r = r[r > 0]
        r = np.sort(r)
        return [float(x) for x in r]
# EVOLVE-BLOCK-END
