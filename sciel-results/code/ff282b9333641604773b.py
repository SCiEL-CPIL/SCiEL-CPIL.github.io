# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Hybrid Thompson Sampling with UCB exploration bonus.
    Combines stochastic sampling (TS) with a deterministic uncertainty
    bonus (UCB) to balance exploration and exploitation robustly.
    The expected score corresponds to a cost-aware UCB with kappa=1.0.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma and cost are positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Draw standard normal random variables for Thompson Sampling
    z = np.random.standard_normal(mu.shape)
    
    # Sample from the posterior distribution N(mu, sigma^2)
    sampled_values = mu + sigma_safe * z
    
    # Base score: Negated sampled value (minimization) divided by cost
    base_score = -sampled_values / cost_safe
    
    # Exploration bonus: Add a term proportional to uncertainty (sigma)
    # This encourages exploring regions with high variance.
    # Using kappa=1.0 aligns the expected score with standard UCB behavior.
    exploration_bonus = 1.0 * sigma_safe / cost_safe
    
    # Combine scores
    scores = base_score + exploration_bonus
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
