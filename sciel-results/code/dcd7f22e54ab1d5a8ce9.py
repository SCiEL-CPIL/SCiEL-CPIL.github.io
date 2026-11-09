"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: RBFs for local flexibility + global trends.
    Combines the local adaptivity of RBFs (current program) with
    global functional forms (polynomial, trig, exp, log) from inspirations.
    
    This gives BFGS a richer search space: RBFs handle local bumps/features,
    while global terms capture overall shape (growth, decay, curvature).
    """
    x = np.asarray(x, dtype=np.float64)
    
    # --- Global trend basis (from inspirations, well-scaled) ---
    x_c = (x - 3.05) / 3.05  # centered and scaled to ~[-1, 1]
    
    # b0: constant
    b0 = np.ones_like(x)
    
    # b1: centered linear
    b1 = x_c
    
    # b2: centered quadratic
    b2 = x_c * x_c
    
    # b3: sin(x)
    b3 = np.sin(x)
    
    # b4: cos(x)
    b4 = np.cos(x)
    
    # b5: exp(x) scaled to be O(1) at center
    b5 = np.exp(x - 3.05) / np.exp(1.5)  # scale down to avoid huge values
    
    # b6: exp(-x) scaled
    b6 = np.exp(-x) * 20.0  # exp(-6)~0.0025, *20 gives ~[0.05, 20]
    
    # b7: log(x) centered
    b7 = np.log(x) - np.log(3.05)
    
    # b8: 1/x centered and scaled
    b8 = (1.0 / x - 1.0 / 3.05) * 3.05
    
    # --- Local RBF basis (from current program, narrow bandwidths) ---
    # b9: narrow RBF at x=3.0 (center of domain) for local feature capture
    b9 = np.exp(-((x - 3.0)**2) / (2 * 0.5**2))
    
    # Stack into basis matrix
    basis = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Compute linear combination
    y = basis @ params
    
    return y

# EVOLVE-BLOCK-END
