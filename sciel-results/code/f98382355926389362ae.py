"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the function using a full quadratic polynomial basis.
    
    Rationale:
    - Previous attempts with rational/ratio bases scored poorly (~9.0), significantly 
      worse than the linear baseline (~1.0). This suggests either numerical instability, 
      poor conditioning of the rational basis, or a mismatch with the target function's 
      structure.
    - A full quadratic polynomial basis (1, x, y, z, x^2, y^2, z^2, xy, xz, yz) is 
      well-conditioned on the domain [0.5, 2] and provides a standard second-order 
      approximation for smooth functions.
    - This basis is linear in the parameters, ensuring the loss landscape is convex, 
      which allows BFGS to reliably find the global minimum.
    - It captures linear trends, curvature, and pairwise interactions, covering a 
      broader class of smooth functions than the previous restricted rational basis.
    """
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    
    # Compute basis elements
    # 1, x, y, z, x^2, y^2, z^2, xy, xz, yz
    pred = c0 * 1.0
    pred += c1 * x
    pred += c2 * y
    pred += c3 * z
    pred += c4 * x * x
    pred += c5 * y * y
    pred += c6 * z * z
    pred += c7 * x * y
    pred += c8 * x * z
    pred += c9 * y * z
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(pred)):
        return np.full_like(pred, 1e30)
        
    return pred
# EVOLVE-BLOCK-END
