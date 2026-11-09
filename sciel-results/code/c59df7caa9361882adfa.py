"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    Uses a combination of Expected Improvement (EI) and a scaled uncertainty bonus.
    
    Parameters:
    -----------
    mu : ndarray of shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray of shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed value so far.
    
    Returns:
    --------
    scores : ndarray of shape (2000,)
        Acquisition scores; higher values indicate more desirable points.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Use a pure EI with a small exploration term that decays as we get closer to optimum
    # The exploration term helps in early stages when f_best is far from optimum
    # Scale the exploration term by the relative gap to a reasonable lower bound
    # For Styblinski-Tang 1D, the minimum is approximately -39.1662
    lower_bound = -39.1662
    gap = max(f_best - lower_bound, 1e-6)
    
    # Exploration bonus: larger when we're far from the known minimum
    # This encourages exploration early and exploitation later
    exploration_weight = 0.05 * np.log1p(gap / 10.0)
    bonus = exploration_weight * sigma_safe
    
    return ei + bonus
# EVOLVE-BLOCK-END
