import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    # Strategy: Use the roots of the generalized Laguerre polynomial
    # L_n^(alpha)(x) with n=12 and alpha=-1/2 as the double-root locations.
    # This is a principled heuristic based on the orthogonality properties
    # of the basis functions used in the definition of g(x).
    
    try:
        # Get roots and weights for L_12^(-0.5)
        roots, _ = roots_genlaguerre(12, -0.5)
        
        # Ensure roots are positive and distinct
        # roots_genlaguerre returns roots in ascending order
        if np.any(roots <= 0) or np.any(np.diff(roots) < 1e-9):
             # Fallback to seed if something went wrong
             return [4.0 * (i + 1) for i in range(12)]
             
        return [float(r) for r in roots]
        
    except Exception:
        # Fallback to seed if scipy fails
        return [4.0 * (i + 1) for i in range(12)]
