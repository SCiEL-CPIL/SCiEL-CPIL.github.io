"""Rewritten acquisition function for cost-aware Bayesian optimization on Levy 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track wall-clock for deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement per unit cost,
    with a hard feasibility gate based on average cost per unit improvement.
    
    Strategy: 
    1. Compute EI for each candidate.
    2. Compute a dynamic threshold: average cost of points that have improved f_best.
    3. Candidates with cost above this threshold get heavily penalized (multiplicative gate).
    4. Final score = EI * gate, where gate is 1.0 if cost <= threshold, else exponential decay.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.ones_like(mu) / (cost + 1e-9)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-6)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute a dynamic cost threshold.
    # We estimate the "average cost per unit improvement" from the current state.
    # Since we don't have full history, we use a heuristic:
    # The cost function is 1 + mean(x_scaled). The minimum possible cost is 1.0.
    # A good strategy is to prefer low-cost points early.
    # Let's use a threshold that starts low and increases as f_best improves.
    # For now, use a simple adaptive threshold based on the 25th percentile of costs
    # weighted by EI, or simply a fixed low threshold that encourages exploitation.
    
    # Alternative structural change: Use UCB with cost-aware penalty but with a different form.
    # Let's try: Score = EI / (cost^alpha) where alpha is adaptive.
    # Or: Score = EI * exp(-cost / T) where T is a temperature.
    
    # The directive suggests: "hard budget-aware gate that forces the search to exploit 
    # the low-cost corner until the GP uncertainty there is exhausted"
    
    # Let's define a threshold as the 30th percentile of the cost array.
    # Points below this threshold get full EI, points above get heavily discounted.
    cost_threshold = np.percentile(cost, 30)
    
    # Create a gate: 1.0 if cost <= threshold, else exp(-(cost - threshold) / (0.1 + cost_threshold))
    gate = np.where(cost <= cost_threshold, 1.0, np.exp(-(cost - cost_threshold) / (0.1 + cost_threshold)))
    
    # Final score
    scores = ei * gate
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
