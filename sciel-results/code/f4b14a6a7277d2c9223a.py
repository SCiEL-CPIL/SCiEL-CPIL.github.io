"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Samples a single realization from the Gaussian posterior for all candidates
    and returns the negative sampled values as scores. This naturally balances
    exploration and exploitation without requiring manual tuning of an
    exploration parameter like beta in UCB.
    
    Args:
        mu: Array of predicted means, shape (2000,).
        sigma: Array of predicted standard deviations, shape (2000,).
        f_best: The best observed objective value so far.
        
    Returns:
        Array of scores, shape (2000,). Higher scores are preferred.
    """
    # Sample from the Gaussian posterior: f ~ N(mu, sigma^2)
    # For each candidate, sample one value from its predictive distribution
    # Use a fixed seed per call to ensure reproducibility within the evaluator's loop
    # but different across calls to get different samples each step
    rng = np.random.default_rng()
    
    # Sample: z_i ~ N(mu_i, sigma_i^2)
    # We want to minimize f, so we want the lowest sample.
    # Score = -sample, so highest score corresponds to lowest sample.
    samples = mu + sigma * rng.standard_normal(size=mu.shape)
    
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
