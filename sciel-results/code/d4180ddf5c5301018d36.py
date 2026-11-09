"""
Strategy: Hybrid Fourier + Polynomial Basis with Reparameterization.

The key insight is to use a basis that:
1. Spans both periodic and polynomial behavior (common in hidden functions).
2. Is well-conditioned enough for BFGS to converge quickly from all-ones.
3. Uses exactly 10 parameters.

We use:
- 1 constant term: 1
- 2 cosine terms: cos(x), cos(2x)
- 2 sine terms: sin(x), sin(2x)
- 3 polynomial terms: x, x^2, x^3
- 1 linear scaling term: x^4 (to capture quartic growth if needed)
- 1 extra high-frequency term: cos(3x)

Total: 10 terms.

The design matrix is:
[1, cos(x), cos(2x), sin(x), sin(2x), x, x^2, x^3, x^4, cos(3x)]

This is a linear model: y = params @ basis(x)

BFGS should handle this well because:
- The basis functions are bounded on [0.1, 6] (except polynomials, but x^4 is manageable).
- The columns are not perfectly collinear.
- The problem is convex (linear in params).

To ensure numerical stability, we clip x to [0.1, 6.0] and handle any NaN/Inf.
We also add a small epsilon to avoid division by zero if needed (not needed here).

We precompute nothing since x is passed in. The computation is O(N) per call, which is fast.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# No precomputation needed; we compute basis on the fly.
# We could precompute if x were fixed, but it's not.

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y using a linear combination of 10 hybrid Fourier + polynomial basis functions.
    params: array of shape (10,)
    x: Input array of shape (N,)
    Returns: array of shape (N,)
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    
    # Clip x to domain to avoid extrapolation issues
    x = np.clip(x, 0.1, 6.0)
    
    # Compute basis functions
    # 1. Constant
    b0 = np.ones_like(x)
    
    # 2. cos(x)
    b1 = np.cos(x)
    
    # 3. cos(2x)
    b2 = np.cos(2.0 * x)
    
    # 4. sin(x)
    b3 = np.sin(x)
    
    # 5. sin(2x)
    b4 = np.sin(2.0 * x)
    
    # 6. x
    b5 = x.copy()
    
    # 7. x^2
    b6 = x * x
    
    # 8. x^3
    b7 = b6 * x
    
    # 9. x^4
    b8 = b7 * x
    
    # 10. cos(3x)
    b9 = np.cos(3.0 * x)
    
    # Combine: y = params[0]*b0 + params[1]*b1 + ... + params[9]*b9
    # Use matrix multiplication for speed
    # Build design matrix: shape (N, 10)
    # This is O(N) memory and compute, which is fine for N=200 or 400.
    
    # For better performance, we can do:
    pred = (
        params[0] * b0 +
        params[1] * b1 +
        params[2] * b2 +
        params[3] * b3 +
        params[4] * b4 +
        params[5] * b5 +
        params[6] * b6 +
        params[7] * b7 +
        params[8] * b8 +
        params[9] * b9
    )
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred

# EVOLVE-BLOCK-END
