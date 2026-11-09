"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.special import genlaguerre, eval_genlaguerre, factorial

def construct():
    """
    Return a list of 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    The score is the upper bound on C4 (lower is better).
    """
    t0 = time.time()
    deadline = 600.0  # safety margin
    
    m = 12
    n = 1
    
    # The task mentions "Fourier sign uncertainty upper bound" and "Cohn-Goncalves Laguerre-polynomial construction"
    # with m=12 double roots and dimension n=1.
    
    # In the Cohn-Goncalves framework for uncertainty, one constructs a polynomial with specific roots
    # to bound the number of sign changes or the uncertainty product.
    
    # For the Fourier uncertainty principle, the optimal bound is related to the variance of the
    # ground state of the harmonic oscillator (Gaussian), which gives a specific constant.
    
    # The "C4" likely refers to a specific constant in the uncertainty inequality.
    # For the standard Heisenberg uncertainty principle in 1D, the optimal constant is 1/2
    # for the product of variances (σ_x * σ_p ≥ 1/2 in natural units, or 1/4π depending on convention).
    
    # However, the task asks for root locations, not the constant directly.
    # The score is "the upper bound on C4" which is a raw value (lower is better).
    
    # Let's think about what the roots should be. In many such constructions, the roots
    # are related to the zeros of Laguerre polynomials or related orthogonal polynomials.
    
    # The seed program returns [4.0, 8.0, 12.0, ..., 48.0]. This is an arithmetic progression.
    # A score of 1.0 means this is the baseline.
    
    # For the Cohn-Goncalves construction, the roots often come from the zeros of
    # Laguerre polynomials L_m^(α)(x) or related functions.
    
    # Let's try using the zeros of the Laguerre polynomial L_12(x) (α=0).
    # The zeros of L_n(x) are all positive and distinct.
    
    # Actually, the task says "double roots", which might mean each root has multiplicity 2
    # in some associated polynomial, or it's a specific construction detail.
    
    # Let's compute the zeros of L_12(x) and see if that gives a better score.
    
    # First, let's try the zeros of the Laguerre polynomial L_12(x).
    try:
        # Use scipy to find zeros of Laguerre polynomial
        from numpy.polynomial.laguerre import laggauss
        # laggauss(n) returns points and weights for Laguerre-Gauss quadrature
        # The points are the zeros of L_n(x)
        points, weights = laggauss(m)
        roots_laguerre = sorted(points)
    except Exception:
        roots_laguerre = [4.0 * (i + 1) for i in range(12)]
    
    # Another possibility: the roots might be related to the zeros of the derivative
    # or a different parameter. Let's also try L_12^(1)(x) (α=1).
    try:
        # For α=1, we can use a different approach
        # The zeros of L_n^(α)(x) can be found using numpy roots of the polynomial
        coeffs = np.poly1d(genlaguerre(m, n))
        roots_alpha1 = np.roots(coeffs.coeffs)
        # Filter for real positive roots
        roots_alpha1 = [r.real for r in roots_alpha1 if abs(r.imag) < 1e-10 and r.real > 0]
        roots_alpha1 = sorted(roots_alpha1)
    except Exception:
        roots_alpha1 = roots_laguerre
    
    # The seed uses [4, 8, 12, ..., 48]. Let's also consider that the optimal roots
    # might be scaled versions of the Laguerre zeros.
    
    # For now, let's return the Laguerre zeros as they are mathematically motivated.
    # The Cohn-Goncalves construction typically uses orthogonal polynomial zeros.
    
    # Check deadline
    if time.time() - t0 > deadline - 10:
        return roots_laguerre
    
    return roots_laguerre
# EVOLVE-BLOCK-END
