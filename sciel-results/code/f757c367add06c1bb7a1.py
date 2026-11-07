"""Symbolic regression: equation(x, params) with 10 free params."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using 10 parameters.
    Uses a mix of scaled polynomials and localized Gaussian basis functions.
    This structure is linear in params, allowing BFGS to converge efficiently
    to the optimal linear combination for the fixed basis.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if x.ndim == 0:
        x = x.reshape(1)

    # Scale x from [0.1, 6] to [-1, 1] for better numerical conditioning
    # Midpoint: 3.05, Half-range: 2.95
    t = (x - 3.05) / 2.95
    
    try:
        # Basis functions: mix of polynomials and trigonometric functions
        # This creates a well-conditioned, linear-in-params model that BFGS
        # can optimize efficiently from the all-ones start.
        
        # Polynomial basis (captures global trends)
        phi0 = 1.0
        phi1 = t
        phi2 = t**2
        phi3 = t**3
        phi4 = t**4
        
        # Trigonometric basis (captures periodic/oscillatory features)
        # Use frequencies that are well-resolved on [-1, 1]
        phi5 = np.sin(np.pi * t)
        phi6 = np.cos(np.pi * t)
        phi7 = np.sin(2.0 * np.pi * t)
        phi8 = np.cos(2.0 * np.pi * t)
        phi9 = np.sin(3.0 * np.pi * t)
        
        result = (p[0] * phi0 +
                  p[1] * phi1 +
                  p[2] * phi2 +
                  p[3] * phi3 +
                  p[4] * phi4 +
                  p[5] * phi5 +
                  p[6] * phi6 +
                  p[7] * phi7 +
                  p[8] * phi8 +
                  p[9] * phi9)
        
        # Check for non-finite values
        if not np.all(np.isfinite(result)):
            return np.zeros_like(x, dtype=np.float64)
            
        return result
    except Exception:
        return np.zeros_like(x, dtype=np.float64)
# EVOLVE-BLOCK-END
