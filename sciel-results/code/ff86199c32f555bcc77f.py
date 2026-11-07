"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of Gaussian RBFs with fixed centers and widths.
    10 parameters act as local weights.
    """
    # Input x is in [0.1, 6]
    # Place 10 centers evenly across the domain
    x_min = 0.1
    x_max = 6.0
    n_centers = 10
    
    # Evenly spaced centers
    centers = np.linspace(x_min, x_max, n_centers)
    
    # Fixed width: choose sigma so that RBFs have some overlap
    # Distance between centers is (6 - 0.1) / 9 = 0.655
    # A sigma of about 0.3-0.5 gives good local coverage with overlap
    # Let's use a sigma that covers roughly 1/3 of the spacing to allow flexibility
    # Or simply divide the domain into 10 bins and set sigma = bin_width / 2
    bin_width = (x_max - x_min) / n_centers
    sigma = bin_width * 0.75  # Slight overlap for smoothness
    
    # Compute RBFs
    # x is shape (N,), centers is shape (10,)
    # diff is shape (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    r2 = diff ** 2
    
    # Gaussian: exp(-r2 / (2 * sigma^2))
    # To avoid underflow/overflow issues, we can compute in log space or just use exp
    # Since sigma is fixed, this is stable
    inv_2_sigma2 = 1.0 / (2.0 * sigma ** 2)
    rbf_values = np.exp(-r2 * inv_2_sigma2)
    
    # Linear combination with params
    # rbf_values shape (N, 10), params shape (10,)
    return rbf_values @ params
# EVOLVE-BLOCK-END
