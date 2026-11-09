"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Probability of Improvement (PoI)
    weighted by inverse cost with a small GP uncertainty bonus.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions at 2000 candidate points.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions at 2000 candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Probability of Improvement: P(f(x) < f_best)
    # Using the standard normal CDF
    z = (f_best - mu) / sigma
    poi = norm.cdf(z)
    
    # Adaptive exploration scale:
    # As f_best improves (gets closer to 0), the scale decreases,
    # reducing the relative weight of uncertainty and favoring exploitation.
    # We use max(abs(f_best), 1e-6) to avoid division by zero and handle early iterations.
    scale = np.maximum(np.abs(f_best), 1e-6)
    
    # Normalized uncertainty: sigma relative to the current best value magnitude
    # This makes the exploration term dimensionless and comparable to PoI
    normalized_uncertainty = sigma / scale
    
    # Combined score: PoI + Exploration, all divided by cost
    # Weight 0.5 balances exploitation (PoI) and exploration (Uncertainty)
    scores = (poi + 0.5 * normalized_uncertainty) / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
