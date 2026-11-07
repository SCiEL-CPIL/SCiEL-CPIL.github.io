# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    - array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB component: balance exploitation and exploration
    # Use adaptive beta based on the spread of predictions
    mu_range = mu.max() - mu.min() if mu.max() > mu.min() else 1.0
    sigma_mean = sigma.mean()
    
    # Beta controls exploration: higher when uncertainty is large relative to scale
    # Start with a moderate exploration weight
    beta = 2.0
    
    # UCB score (for minimization, we want low mu + beta * sigma)
    # Transform to a "desirability" where higher is better
    ucb = -mu + beta * sigma
    
    # Normalize UCB to comparable scale with EI
    ucb_min = ucb.min()
    ucb_max = ucb.max()
    ucb_range = ucb_max - ucb_min if ucb_max > ucb_min else 1.0
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Normalize EI to comparable scale
    ei_max = ei.max() if ei.max() > 0 else 1.0
    ei_norm = ei / ei_max
    
    # Combine EI and UCB with a blend
    # Weight EI more heavily for exploitation, UCB for exploration
    blend_weight = 0.6  # 60% EI, 40% UCB
    combined = blend_weight * ei_norm + (1 - blend_weight) * ucb_norm
    
    # Cost-aware weighting: cheaper evaluations should be preferred
    # Use a power-based cost penalty: score / cost^alpha
    # alpha=1 gives linear cost penalty, alpha>1 gives stronger preference for cheap points
    alpha = 1.5
    cost_safe = np.maximum(cost, 1e-6)
    cost_penalty = cost_safe ** alpha
    
    # Apply cost-aware scoring
    scores = combined / cost_penalty
    
    # Add a small exploitation bonus for points near the current best
    # This helps maintain focus on promising regions
    # Use a Gaussian kernel centered at the point with lowest mu
    mu_min_idx = np.argmin(mu)
    mu_min_val = mu[mu_min_idx]
    
    # Distance in prediction space (approximate)
    dist = np.abs(mu - mu_min_val)
    # Scale distance by the typical sigma
    dist_scaled = dist / (sigma_mean + 1e-8)
    # Gaussian kernel
    exploitation_bonus = np.exp(-0.5 * dist_scaled ** 2) * 0.1
    
    # Add exploitation bonus
    scores = scores + exploitation_bonus
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
