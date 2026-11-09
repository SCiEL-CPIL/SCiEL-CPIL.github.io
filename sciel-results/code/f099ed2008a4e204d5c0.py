"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of 1D Sphere function.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the lowest observed function value.
        
    Returns:
        (2000,) array of acquisition scores. Higher is better.
    """
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive Switching Strategy:
    # In the early stages (f_best >= 1.0), use EI to balance exploration and exploitation.
    # In the late stages (f_best < 1.0), switch to Post-Mean Maximization (PMM).
    # For minimization, PMM corresponds to minimizing the GP mean.
    # This avoids the conservative bias of EI near the optimum, allowing for
    # faster convergence to the exact minimum (0).
    if f_best < 1.0:
        # We want to pick the point with the lowest mu.
        # Score should be high for low mu.
        return -mu
    else:
        return ei
# EVOLVE-BLOCK-END
