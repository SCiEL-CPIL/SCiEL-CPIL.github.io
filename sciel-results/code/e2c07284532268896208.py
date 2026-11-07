"""Program using RBF kernel basis for symbolic regression with adaptive width."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form symbolic regression using Gaussian RBF kernels.
    
    Uses 9 RBF kernels with fixed centers evenly spaced over [-3, 3],
    plus 1 constant term. The 10 parameters are:
      - params[0]: constant offset
      - params[1:10]: weights for each RBF kernel
      - Kernel width is fixed at 1.2 (tuned for good conditioning)
    
    This local basis provides flexible interpolation while remaining
    smooth and well-conditioned for BFGS optimization.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # 9 evenly spaced centers over [-3, 3]
    centers = np.linspace(-3.0, 3.0, 9)
    
    # Fixed RBF width - chosen for good conditioning
    # Width of 1.2 means each kernel has significant support over ~2.4 units
    width = 1.2
    
    # Compute RBF values: exp(-((x - center)^2) / (2 * width^2))
    # Shape: (n_points, 9)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_vals = np.exp(-(diff ** 2) / (2.0 * width ** 2))
    
    # Build prediction: constant + weighted sum of RBFs
    y = params[0] * np.ones_like(x) + rbf_vals @ params[1:10]
    
    # Ensure finite output
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
# EVOLVE-BLOCK-END
