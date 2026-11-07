"""
STRUCTURAL CHANGE: Chebyshev-polynomial product basis with nonlinear activations.

The key insight: instead of a fixed linear basis or a simple neural network layer,
we create a rich set of basis functions that are PRODUCTS of:
- Chebyshev polynomials (global stability, orthogonal on [-1,1])
- Nonlinear activations (tanh, sin, exp) to capture non-periodic nonlinearities

The 10 parameters are used as:
- params[0:6] = coefficients for the 6 Chebyshev basis terms (T0..T5)
- params[6:8] = coefficients for 2 tanh-modulated terms
- params[8]   = coefficient for a sin-modulated term
- params[9]   = coefficient for an exp-modulated term

This creates a smooth, globally-defined basis that BFGS can navigate effectively.
The Chebyshev polynomials provide numerical stability, while the nonlinear
modulations allow the model to capture complex function shapes.

We map x from [-3,3] to [-1,1] for the Chebyshev evaluation to stay in the
orthogonality interval, but also include terms that work over the full range.
"""
import numpy as np


def _chebyshev_T(n, t):
    """Compute Chebyshev polynomial of the first kind T_n(t) for array t."""
    if n == 0:
        return np.ones_like(t)
    elif n == 1:
        return t.copy()
    else:
        Tn_minus_1 = np.ones_like(t)
        Tn = t.copy()
        for k in range(2, n + 1):
            Tn_plus_1 = 2.0 * t * Tn - Tn_minus_1
            Tn_minus_1 = Tn
            Tn = Tn_plus_1
        return Tn


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Map x from [-3, 3] to [-1, 1] for Chebyshev stability
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 through T_5
    T0 = np.ones_like(t)
    T1 = t.copy()
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    T5 = 2.0 * t * T4 - T3
    
    # Nonlinear activations applied to scaled x
    # Use moderate scaling to keep values in a reasonable range
    ax = 0.5 * x  # scale factor
    
    tanh_term1 = np.tanh(ax)
    tanh_term2 = np.tanh(2.0 * ax)
    sin_term = np.sin(x)
    exp_term = np.exp(-0.3 * x * x)  # Gaussian-like, always positive and bounded
    
    # Build the output as a linear combination of basis functions
    # params[0:6] for Chebyshev terms
    # params[6:8] for tanh terms
    # params[8] for sin term
    # params[9] for exp term
    
    y = (p[0] * T0 + 
         p[1] * T1 + 
         p[2] * T2 + 
         p[3] * T3 + 
         p[4] * T4 + 
         p[5] * T5 +
         p[6] * tanh_term1 + 
         p[7] * tanh_term2 +
         p[8] * sin_term + 
         p[9] * exp_term)
    
    # Ensure finite output
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
