"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy: Use a diverse basis of 10 features that are linear in params.
# The key insight is to choose features that span a wide range of functional forms:
# - Polynomial terms (up to degree 2-3)
# - Trigonometric terms (for periodic behavior)
# - Rational terms (for asymptotic behavior)
# - Exponential-like terms (via scaled inputs)
#
# Since inputs are in [0.5, 2], we can safely use:
# - sin, cos (bounded, smooth)
# - 1/x, 1/y, 1/z (smooth on [0.5, 2])
# - x^2, y^2, z^2 (polynomial)
# - x*y, x*z, y*z (interactions)
# - x*y*z (three-way interaction)
#
# We need to be careful about feature scaling so BFGS works well.
# Normalize some features to have O(1) magnitude on the input domain.

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression to recover an unknown function of 3 inputs.
    Uses a linear-in-params model with 10 diverse basis features.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Build 10 diverse features
    # Polynomial and interaction terms
    f0 = np.ones_like(x)          # constant
    f1 = x - 1.25                 # centered x (mean of [0.5,2] is 1.25)
    f2 = y - 1.25                 # centered y
    f3 = z - 1.25                 # centered z
    f4 = (x - 1.25) * (y - 1.25)  # centered xy interaction
    f5 = (x - 1.25) * (z - 1.25)  # centered xz interaction
    f6 = (y - 1.25) * (z - 1.25)  # centered yz interaction
    f7 = (x - 1.25) * (y - 1.25) * (z - 1.25)  # centered xyz
    
    # Non-polynomial terms for broader coverage
    f8 = np.sin(x) + np.sin(y) + np.sin(z)  # trigonometric component
    f9 = 1.0/(x + y + z)                     # rational component
    
    # Linear combination
    pred = (params[0] * f0 + 
            params[1] * f1 + 
            params[2] * f2 + 
            params[3] * f3 + 
            params[4] * f4 + 
            params[5] * f5 + 
            params[6] * f6 + 
            params[7] * f7 + 
            params[8] * f8 + 
            params[9] * f9)
    
    return pred
# EVOLVE-BLOCK-END
