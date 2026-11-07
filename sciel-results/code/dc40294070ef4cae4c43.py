import numpy as np
import time

_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation for symbolic regression with 1 input.
    Uses a Radial Basis Function (RBF) expansion with 10 fixed Gaussian centers.
    The parameters are the weights of the Gaussians.
    This local basis is often more robust for arbitrary smooth functions 
    on a bounded interval than global polynomials, avoiding Runge's phenomenon
    and providing better conditioning for BFGS.
    """
    # Deadline guard
    if time.time() - _START_TIME > 145.0:
        return np.zeros_like(x)
    
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), constant_values=0.0)
    else:
        p = p[:10]
        
    # Define 10 fixed centers evenly spaced in [-2, 2]
    # Centers: -2.0, -1.555, ..., 2.0
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed standard deviation for the Gaussian RBFs
    # A width of 0.5 provides good local coverage with overlap
    sigma = 0.5
    
    # Compute the RBF matrix
    # x is (N,), centers is (10,)
    # diff is (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian: exp(-diff^2 / (2 * sigma^2))
    # Pre-compute constant factor
    inv_2_sigma2 = 1.0 / (2.0 * sigma * sigma)
    
    basis = np.exp(-diff * diff * inv_2_sigma2)
    
    # Linear combination: y = basis @ p
    # basis is (N, 10), p is (10,)
    result = basis @ p
    
    # Ensure finite
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
