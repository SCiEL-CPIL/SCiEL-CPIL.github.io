import scipy.special
import numpy as np

def construct():
    # Use the roots of the generalized Laguerre polynomial L_12^(-1/2)(x)
    # This aligns with the orthogonality weight and the specific alpha of the problem.
    # The asymptotic Bessel order for alpha=-1/2 is 0, so J0 zeros are a good asymptotic approximation,
    # but the exact Laguerre roots for n=12 should be better for the finite case.
    try:
        # roots_genlaguerre(n, alpha) returns the roots of L_n^(alpha)
        # n=12, alpha=-0.5
        roots, weights = scipy.special.roots_genlaguerre(12, -0.5)
        # The roots are real, positive, and distinct.
        # Convert to list of floats
        z = [float(r) for r in roots]
        return z
    except Exception:
        # Fallback to J0 zeros squared if something fails
        try:
            j0_zeros = scipy.special.jn_zeros(0, 12)
            z = [float(x**2) for x in j0_zeros]
            return z
        except Exception:
            return [4.0 * (i + 1) for i in range(12)]
