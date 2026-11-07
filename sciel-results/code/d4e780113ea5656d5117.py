"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for the acceleration of a nonlinear oscillator.
    
    The model uses a polynomial basis in position (x) and velocity (v) up to 3rd order.
    This covers Duffing-type stiffness, Van der Pol-type nonlinear damping, and 
    various coupling terms.
    
    Terms:
    1. x
    2. x^2
    3. x^3
    4. v
    5. v^2
    6. v^3
    7. x*v
    8. x^2*v
    9. x*v^2
    10. 1 (constant)
    """
    # Extract parameters
    p = params
    
    # Compute basis functions
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Construct the linear combination
    # Note: We do not use 't' explicitly in the basis to focus on autonomous dynamics 
    # as suggested by the directive, assuming the trajectory t, x, v implicitly encodes 
    # the time evolution or the system is autonomous. If the system is driven, 
    # the driving might be captured by the constant or other terms if it's periodic 
    # and correlates with x/v, or the model might be less accurate for non-autonomous 
    # driven systems without explicit t terms. However, the directive specifically 
    # suggests excluding explicit time-dependent driving terms to capture intrinsic dynamics.
    
    term1 = x
    term2 = x2
    term3 = x3
    term4 = v
    term5 = v2
    term6 = v3
    term7 = x * v
    term8 = x2 * v
    term9 = x * v2
    term10 = 1.0
    
    return (p[0] * term1 +
            p[1] * term2 +
            p[2] * term3 +
            p[3] * term4 +
            p[4] * term5 +
            p[5] * term6 +
            p[6] * term7 +
            p[7] * term8 +
            p[8] * term9 +
            p[9] * term10)
# EVOLVE-BLOCK-END
