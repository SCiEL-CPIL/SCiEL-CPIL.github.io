# EVOLVE-BLOCK-START
import numpy as np
import time

# Fixed centers for the Radial Basis Function (RBF) representation
# Using 10 centers to utilize all 10 parameters as weights
_CENTERS = np.linspace(-2.0, 2.0, 10)

# Fixed width parameter for the Gaussian RBF kernel
# w = 1 / sigma^2. 
# sigma ~ 0.3 provides a good balance between locality and smoothness for 10 centers in [-2, 2]
# Distance between centers is ~0.44. 
# At 1 sigma, value is 0.37. At 1.47 sigma (distance), value is 0.11.
# This ensures significant overlap for smooth interpolation but distinct centers.
_WIDTH = 1.0 / (0.3 ** 2)  # ~11.11

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Gaussian Radial Basis Function (RBF) model.
    The 10 parameters are the weights for 10 fixed Gaussian centers.
    This provides a flexible, smooth, non-parametric representation that is
    linear in the parameters, ensuring robust convergence for BFGS.
    """
    # Ensure inputs are 1D arrays for consistency
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Check for non-finite params to avoid crashes
    if not np.all(np.isfinite(params)):
        return np.zeros_like(x)
        
    try:
        # Compute squared distances from each x to each center
        # x shape (N,), centers shape (10,)
        # dists_sq shape (N, 10)
        dists_sq = (x[:, np.newaxis] - _CENTERS[np.newaxis, :]) ** 2
        
        # Compute Gaussian RBF values: exp(-w * dists_sq)
        # rbf shape (N, 10)
        rbf_values = np.exp(-_WIDTH * dists_sq)
        
        # Linear combination of RBF values with params
        # y_pred shape (N,)
        y_pred = rbf_values @ params
        
        # Check for non-finite results
        if not np.all(np.isfinite(y_pred)):
             return np.zeros_like(x)
             
        return y_pred
    except Exception:
        # Fallback to a simple linear model if RBF evaluation fails
        return x * params[0] + params[1]
# EVOLVE-BLOCK-END
