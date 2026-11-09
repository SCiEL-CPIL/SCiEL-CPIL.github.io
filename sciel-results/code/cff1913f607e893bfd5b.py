"""Improved symbolic regression model for aluminium rod stress-strain."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters model with fixed non-linear basis functions.
    
    This structure ensures the optimization problem is convex (linear least squares),
    allowing BFGS to find the global minimum reliably. The non-linear behavior
    (yield, hardening) is captured by the choice of basis functions (polynomials
    and tanh), while the 10 parameters are simple weights.
    
    Basis functions:
      1. eps
      2. eps^2
      3. eps^3
      4. eps^4
      5. T * eps
      6. T * eps^2
      7. T^2 * eps
      8. tanh(5 * eps)
      9. tanh(5 * eps) * T
      10. tanh(5 * eps) * T^2
    """
    # Ensure inputs are arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Extract weights
    w = params
    
    # Compute basis functions
    eps = strain
    T = temp
    
    # Polynomial terms
    b1 = eps
    b2 = eps * eps
    b3 = b2 * eps
    b4 = b3 * eps
    
    # Temperature-modulated polynomial terms
    b5 = T * eps
    b6 = T * b2
    b7 = T * T * eps
    
    # Non-linear saturation terms (fixed slope 5.0 to capture yield in ~0.1 strain)
    # Clip to avoid overflow in tanh (though tanh is stable, good practice)
    tanh_arg = 5.0 * eps
    tanh_val = np.tanh(tanh_arg)
    
    b8 = tanh_val
    b9 = tanh_val * T
    b10 = tanh_val * T * T
    
    # Linear combination
    prediction = (w[0] * b1 + w[1] * b2 + w[2] * b3 + w[3] * b4 +
                  w[4] * b5 + w[5] * b6 + w[6] * b7 +
                  w[7] * b8 + w[8] * b9 + w[9] * b10)
    
    return prediction
# EVOLVE-BLOCK-END
