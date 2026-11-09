# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track time since first call to stay within budget.
_t_start = None
_DEADLINE_FRACTION = 0.92  # stop at 92% of 330s to be safe

def acquisition(mu, sigma, f_best, cost):
    global _t_start
    if _t_start is None:
        _t_start = time.time()
    
    # Safety: if we're running out of time, return a simple but valid score
    elapsed = time.time() - _t_start
    if elapsed > 300.0:
        # Return a valid finite array immediately
        z = (f_best - mu) / (sigma + 1e-9)
        return ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / (cost + 1e-9)
    
    # Cost-aware acquisition: combine Expected Improvement with a cost penalty.
    # We use a modified EI that accounts for the cost structure.
    # The cost is 1 + mean(x_scaled), so cheaper points (lower x) are favored.
    # But since we don't have x, we use the provided cost array.
    
    # Standard EI
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    # But also add a small exploration bonus for high uncertainty
    # to avoid getting stuck in local optima of rastrigin
    
    # Use a weighted combination: mostly EI/cost, plus a small UCB-like term
    # to encourage exploration in uncertain regions
    exploration_weight = 0.1
    ucb = mu + exploration_weight * sigma  # lower is better, so we subtract for minimization
    # Actually for minimization, we want low mu and high sigma (exploration)
    # Let's use: score = EI/cost + small * sigma/cost (exploration bonus)
    
    score = ei / (cost + eps) + 0.05 * sigma / (cost + eps)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
