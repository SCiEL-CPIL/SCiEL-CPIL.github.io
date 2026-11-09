"""Symbolic regression with 3 inputs using a non-linear feature basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a non-linear feature basis.
    
    This approach uses logarithmic, reciprocal, and interaction terms,
    which are effective for capturing power-law, exponential, and
    asymptotic behaviors that pure polynomials may approximate poorly.
    
    Basis terms (10 parameters):
    1. Constant: 1
    2. Logarithmic: log(x)
    3. Logarithmic: log(y)
    4. Logarithmic: log(z)
    5. Interaction: x * y
    6. Interaction: y * z
    7. Interaction: z * x
    8. Reciprocal: 1/x
    9. Reciprocal: 1/y
    10. Reciprocal: 1/z
    """
    p = params
    
    # Ensure finite values in params
    if not np.all(np.isfinite(p)):
        return np.ones_like(x)
    
    # Inputs are in [0.5, 2], so log and reciprocal are safe and well-conditioned.
    # Clip to avoid any edge case issues if inputs were slightly outside, though contract says [0.5, 2].
    x_safe = np.clip(x, 1e-6, None)
    y_safe = np.clip(y, 1e-6, None)
    z_safe = np.clip(z, 1e-6, None)
    
    # Build the non-linear basis
    term0 = 1.0
    term1 = np.log(x_safe)
    term2 = np.log(y_safe)
    term3 = np.log(z_safe)
    term4 = x_safe * y_safe
    term5 = y_safe * z_safe
    term6 = z_safe * x_safe
    term7 = 1.0 / x_safe
    term8 = 1.0 / y_safe
    term9 = 1.0 / z_safe
    
    result = (p[0] * term0 +
              p[1] * term1 +
              p[2] * term2 +
              p[3] * term3 +
              p[4] * term4 +
              p[5] * term5 +
              p[6] * term6 +
              p[7] * term7 +
              p[8] * term8 +
              p[9] * term9)
    
    # Ensure output is finite
    if not np.all(np.isfinite(result)):
        return np.ones_like(x)
        
    return result
# EVOLVE-BLOCK-END
