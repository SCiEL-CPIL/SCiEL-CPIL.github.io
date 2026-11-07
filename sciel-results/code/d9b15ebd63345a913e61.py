import numpy as np
from scipy.optimize import minimize

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a radial basis function (RBF) representation.
    
    The 10 parameters are organized as:
    - params[0:3]: x-coordinates of 3 Gaussian centers
    - params[3:6]: v-coordinates of 3 Gaussian centers
    - params[6:7]: width (sigma) for all Gaussians
    - params[7:10]: coefficients for the 3 RBF terms
    
    This provides a more flexible local approximation than global polynomials.
    """
    # Extract parameters
    cx = params[0:3]  # 3 x-centers
    cv = params[3:6]  # 3 v-centers
    sigma = params[6]  # shared width
    coeffs = params[7:10]  # 3 coefficients
    
    # Safety: avoid division by zero or negative sigma
    sigma = np.abs(sigma) + 1e-6
    
    # Compute squared distances to each center
    # x and v are arrays of shape (n,)
    # cx, cv are arrays of shape (3,)
    
    # For each center i, compute: exp(-((x - cx_i)^2 + (v - cv_i)^2) / (2 * sigma^2))
    
    # Vectorized computation
    # dx = x[:, np.newaxis] - cx[np.newaxis, :]  # shape (n, 3)
    # dv = v[:, np.newaxis] - cv[np.newaxis, :]  # shape (n, 3)
    # dist_sq = dx**2 + dv**2  # shape (n, 3)
    # rbf_vals = np.exp(-dist_sq / (2 * sigma**2))  # shape (n, 3)
    # result = rbf_vals @ coeffs  # shape (n,)
    
    dx = x[:, np.newaxis] - cx[np.newaxis, :]
    dv = v[:, np.newaxis] - cv[np.newaxis, :]
    dist_sq = dx * dx + dv * dv
    rbf_vals = np.exp(-dist_sq / (2.0 * sigma * sigma))
    
    return rbf_vals @ coeffs
