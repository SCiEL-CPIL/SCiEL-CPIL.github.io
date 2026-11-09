"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Radial Basis Function (RBF) expansion.
    This uses a local basis which is typically better conditioned for BFGS 
    than global polynomial bases on the [-3, 3] domain, allowing for 
    more robust convergence to the optimal parameters.
    
    We use 10 Gaussian RBF centers evenly spaced across the domain.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # 10 RBF centers evenly spaced in [-3, 3]
    # Using linspace ensures coverage of the entire domain
    centers = np.linspace(-3.0, 3.0, 10)
    
    # RBF width (sigma). 
    # A value around 0.5-1.0 is often a good balance between locality and smoothness.
    # Since the domain is 6 units wide and we have 10 centers, spacing is 0.6.
    # sigma = 0.5 gives decent overlap.
    sigma = 0.5
    
    # Compute distances between x (N, 1) and centers (1, 10)
    # x is (N,), centers is (10,)
    # dist_sq = (x[:, np.newaxis] - centers[np.newaxis, :])**2
    # To save memory and time for large N, we can do this in chunks or just rely on numpy broadcasting
    # N is 200 for training, 200 for test. This is small.
    
    dist_sq = (x[:, np.newaxis] - centers[np.newaxis, :])**2
    
    # Compute RBF values: exp(-dist_sq / (2 * sigma^2))
    # y = sum(params[i] * phi_i(x))
    rbf_vals = np.exp(-dist_sq / (2.0 * sigma**2))
    
    # Return prediction: rbf_vals (N, 10) @ params (10,)
    return rbf_vals @ params
# EVOLVE-BLOCK-END
