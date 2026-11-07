"""Symbolic regression: recover hidden univariate f on [-3,3] with 10 free params.

KEY INSIGHT: The seed model is non-linear in params (amplitude * sin(freq*x) etc.)
with abs() kinks, making BFGS struggle. A model LINEAR in the 10 params has a
convex quadratic MSE objective with a unique global minimum, so BFGS converges
reliably to the least-squares solution from the all-ones start.

We use a rich, bounded, smooth basis of 10 functions spanning:
  - polynomials up to degree 4 (even/odd)
  - first and second harmonics (sin/cos)
  - a saturating odd function (tanh)
  - a rational even bump (Lorentzian)

All basis values are bounded on [-3,3], ensuring numerical stability.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-params model: f(x) = sum_i p_i * phi_i(x)
    
    Basis functions (10 total):
      phi_0 = 1
      phi_1 = x
      phi_2 = x^2
      phi_3 = x^3
      phi_4 = x^4
      phi_5 = sin(x)
      phi_6 = cos(x)
      phi_7 = sin(2x)
      phi_8 = cos(2x)
      phi_9 = tanh(x)
    
    This basis is diverse: polynomials capture smooth trends, trigonometric
    terms capture periodicity, and tanh captures saturation. All are bounded
    on [-3,3] for numerical stability.
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    
    # Compute basis functions
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    phi = np.column_stack([
        np.ones_like(x),       # phi_0
        x,                      # phi_1
        x2,                     # phi_2
        x3,                     # phi_3
        x4,                     # phi_4
        np.sin(x),              # phi_5
        np.cos(x),              # phi_6
        np.sin(2.0 * x),        # phi_7
        np.cos(2.0 * x),        # phi_8
        np.tanh(x),             # phi_9
    ])
    
    # Linear combination: f(x) = phi @ p
    y = phi @ p
    
    return y
# EVOLVE-BLOCK-END
