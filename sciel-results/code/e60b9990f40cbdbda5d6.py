"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Hybrid acquisition: blend of Expected Improvement (EI) and UCB,
    with cost-awareness and adaptive exploration.
    
    The idea:
    - EI exploits regions where we expect improvement over f_best
    - UCB explores regions with high uncertainty
    - We blend them with a weight that shifts from exploration to exploitation
      as the budget is consumed (using cost as a proxy for progress)
    - Cost-awareness: divide by cost to prefer cheaper evaluations
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from GP
    sigma : array-like, shape (2000,)
        Standard deviation predictions from GP
    f_best : float
        Best observed function value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB term: exploit + explore
    # Use a beta parameter that balances exploitation and exploration
    # As cost accumulates (proxy for iterations), we shift more toward exploitation
    mean_cost = np.mean(cost_safe)
    # Normalize cost to get a progress indicator
    progress = np.clip(mean_cost / 30.0, 0.0, 1.0)  # 30 is the budget
    
    # Beta: starts high (exploratory), decreases as we progress
    beta = 2.0 * (1.0 - 0.7 * progress)
    
    # UCB score
    ucb = -mu + beta * sigma_safe
    
    # Normalize EI and UCB to comparable scales
    # EI is in objective units, UCB is also in objective units (since mu is)
    
    # Blend: more EI when we have good estimates, more UCB early on
    # Use a sigmoid-like blend based on progress
    ei_weight = 0.3 + 0.7 * progress  # starts at 0.3, goes to 1.0
    ucb_weight = 1.0 - ei_weight * 0.5  # complementary but not strictly
    
    # Actually, let's use a cleaner blend:
    # score = alpha * EI + (1-alpha) * UCB, where alpha increases with progress
    alpha = 0.4 + 0.6 * progress
    
    combined = alpha * ei + (1.0 - alpha) * ucb
    
    # Cost-aware: divide by cost to prefer cheaper points
    scores = combined / cost_safe
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Clip extreme values to avoid numerical issues
    scores = np.clip(scores, -1e6, 1e6)
    
    return scores
# EVOLVE-BLOCK-END
