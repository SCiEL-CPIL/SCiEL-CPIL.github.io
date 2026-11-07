"""Cost-aware acquisition for Powell 4D BO with non-uniform evaluation costs."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for time tracking
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Powell 4D BO.
    
    Parameters:
    - mu: (2000,) array of GP mean predictions
    - sigma: (2000,) array of GP std predictions  
    - f_best: scalar, best observed value so far
    - cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    - (2000,) array of acquisition scores (higher = better)
    """
    # Time guard: if we've been running too long, return safe fallback
    elapsed = _get_elapsed()
    if elapsed > 280:  # Safety margin before 300s limit
        # Return simple EI/cost as fallback
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)
    
    n = len(mu)
    
    # Compute expected improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Knowledge gradient approximation: weight by uncertainty in the minimum
    # Use a softmax-like weighting to focus on promising regions
    # Scale EI to create a probability distribution over candidates
    ei_shifted = ei - ei.max()  # Shift so max is 0
    # Use a temperature to control exploration
    # Higher temperature = more exploration
    temp = 0.1 + 0.5 * min(1.0, np.mean(sigma) / (np.std(mu) + 1e-9))
    weights = np.exp(ei_shifted / temp)
    weights /= weights.sum()
    
    # Cost-aware scoring: we want high EI but low cost
    # Use a multiplicative penalty for cost
    # The cost is in [1, 2] since x is in [0,1]^4 and cost = 1 + mean(x)
    # Normalize cost to [0, 1] for the penalty
    cost_norm = (cost - cost.min()) / (cost.max() - cost.min() + 1e-9)
    
    # Adaptive exploration: increase exploration as we get more confident
    # (i.e., as sigma decreases in the best region)
    sigma_ratio = np.mean(sigma) / (np.max(mu) - np.min(mu) + 1e-9)
    explore_rate = 0.3 + 0.4 * min(1.0, sigma_ratio)
    
    # Combine: EI per unit cost with exploration bonus
    # Base score: EI / cost
    base_score = ei / (cost + 1e-9)
    
    # Exploration bonus: prefer candidates with high uncertainty
    # that are not too expensive
    exploration_bonus = explore_rate * sigma / (cost + 1e-9)
    
    # Diversity bonus: prefer candidates that are in under-explored regions
    # Approximate by using the GP variance as a proxy
    diversity_bonus = 0.1 * sigma / (cost + 1e-9)
    
    # Final score
    score = base_score + exploration_bonus + diversity_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
