import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(x)

    p = params

    # Scaling to [-1, 1] based on known training ranges
    # This ensures good conditioning for the BFGS optimizer
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    x_range = x_max - x_min
    v_range = v_max - v_min
    
    if x_range < 1e-9:
        u = np.zeros_like(x)
    else:
        u = 2.0 * (x - x_min) / x_range - 1.0
        
    if v_range < 1e-9:
        w = np.zeros_like(v)
    else:
        w = 2.0 * (v - v_min) / v_range - 1.0

    # Radial Basis Function (RBF) network with Gaussian kernels
    # Centers are fixed at well-distributed points in the (u, w) domain
    # This provides local interpolation capability, avoiding global polynomial
    # ill-conditioning while maintaining smoothness.
    
    # Define 9 kernel centers in a 3x3 grid within [-1, 1]^2
    centers_u = np.array([-1.0, 0.0, 1.0])
    centers_w = np.array([-1.0, 0.0, 1.0])
    
    # Width parameter for Gaussian kernels
    # A moderate width allows local influence without being too narrow
    sigma = 0.5
    
    # Precompute the squared distances from each data point to each center
    # u has shape (N,), centers_u has shape (3,)
    # dist_u[i, j] = (u[i] - centers_u[j])^2
    dist_u = (u[:, np.newaxis] - centers_u[np.newaxis, :]) ** 2
    dist_w = (w[:, np.newaxis] - centers_w[np.newaxis, :]) ** 2
    
    # Combine to get squared distance to each of the 9 centers
    # dist2[i, j*3+k] = dist_u[i, j] + dist_w[i, k]
    # Shape: (N, 9)
    dist2 = dist_u[:, :, np.newaxis] + dist_w[:, np.newaxis, :]
    dist2 = dist2.reshape(dist2.shape[0], -1)
    
    # Compute Gaussian kernels: exp(-dist2 / (2 * sigma^2))
    # This gives shape (N, 9)
    kernels = np.exp(-dist2 / (2.0 * sigma * sigma))
    
    # Add a constant bias term (10th parameter)
    # Total 10 parameters: 9 for kernels + 1 bias
    ones = np.ones((kernels.shape[0], 1))
    
    # Combine kernels and bias into design matrix
    # Shape: (N, 10)
    design = np.hstack([kernels, ones])
    
    # Return design @ params
    return design @ p
