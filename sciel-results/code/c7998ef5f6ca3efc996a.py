"""Cost-aware acquisition for BO on four benchmark functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 295.0  # Safety margin before 300s/330s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        return np.zeros_like(mu)
    
    # Ensure sigma is positive for sampling
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Thompson Sampling:
    # Sample a single function value y from the posterior predictive distribution
    # y ~ N(mu, sigma^2)
    # Since we want to minimize, low y is good.
    # We sample 2000 values.
    samples = mu + sigma_safe * np.random.randn(len(mu))
    
    # 2. Cost-aware scaling
    # The cost is 1 + mean(x_scaled).
    # We want to favor low cost.
    # Instead of dividing the score by cost, we can subtract a cost penalty or divide.
    # Dividing is equivalent to maximizing (Benefit / Cost).
    # Let's try a hybrid: ( -samples + exploration_bonus ) / cost^alpha
    
    cost_safe = np.maximum(cost, 1e-9)
    
    # 3. Adaptive Exploration-Exploitation
    # In the early stages or when uncertainty is high, we should explore more.
    # We can estimate the "regret" or distance from the best found so far.
    # If f_best is far from the expected minimum (which we don't know, but we know f_best),
    # and sigma is large, we explore.
    
    # A simple heuristic for exploration weight:
    # If the relative uncertainty (sigma / |f_best|) is high, we trust the samples less
    # and add a direct exploration term based on sigma.
    
    f_best_abs = np.maximum(np.abs(f_best), 1e-9)
    rel_uncertainty = sigma_safe / f_best_abs
    
    # Clip the relative uncertainty to avoid extreme weights
    rel_uncertainty_clipped = np.clip(rel_uncertainty, 0.0, 3.0)
    
    # Weight for exploration (sigma) vs exploitation (sampled value)
    # When rel_uncertainty is high, we want to explore more (higher weight on sigma).
    # When rel_uncertainty is low, we want to exploit (lower weight on sigma).
    # Let's define exploration_weight such that it increases with rel_uncertainty.
    # exploration_weight = rel_uncertainty_clipped / (1.0 + rel_uncertainty_clipped)
    # This ranges from 0 to 0.75.
    
    # Alternatively, just use the sample directly. Thompson sampling already balances this.
    # But we need to handle the cost.
    
    # The "value" of a point in Thompson sampling for minimization is -sample.
    # Higher is better.
    value = -samples
    
    # Add an exploration bonus that is proportional to sigma and inversely proportional to cost?
    # Or just divide the value by cost?
    
    # Let's try: Score = ( -samples + beta * sigma_safe ) / cost_safe^alpha
    # Where beta controls the exploration bonus.
    # If beta is 0, it's pure exploitation (greedy on samples).
    # If beta is large, it's pure exploration (UCB-like).
    
    # Adaptive beta:
    # If we have found a very good point (f_best close to 0), we should exploit more (lower beta).
    # If f_best is poor, we should explore more (higher beta).
    
    # Let's set beta based on the relative uncertainty.
    # beta = 0.5 * rel_uncertainty_clipped
    
    beta = 0.5 * rel_uncertainty_clipped
    
    # Cost exponent:
    # Lower alpha means we are less sensitive to cost.
    # Higher alpha means we strongly prefer cheap points.
    # Since cost is between 1 and 2, the impact is mild.
    # Let's use alpha = 1.0 for a direct cost-efficiency ratio.
    alpha = 1.0
    
    cost_weight = cost_safe ** alpha
    
    score = (value + beta * sigma_safe) / cost_weight
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
