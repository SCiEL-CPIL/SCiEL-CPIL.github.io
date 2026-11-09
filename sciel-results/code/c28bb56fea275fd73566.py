# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines Cost-Adjusted Probability of Improvement (PoI) with
    a scaled Expected Improvement (EI) term and an exploration bonus.
    
    Score = w_poi * (PoI / cost) + w_ei * (EI / scale) + w_explore * (sigma / scale)
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for EI and PoI (minimization)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -30, 30)
    
    # Probability of Improvement (minimization): P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Scale for EI normalization: use max of |f_best|, mean EI, and small epsilon
    scale = max(abs(f_best), np.mean(np.maximum(ei, 1e-10)), 1e-6)
    
    # Cost-adjusted PoI: prioritize high likelihood of success per unit cost
    poi_cost = pi / cost_safe
    
    # Scaled EI: magnitude of expected improvement relative to problem scale
    ei_scaled = ei / scale
    
    # Exploration bonus: encourages visiting high-uncertainty regions
    exploration_bonus = 0.15 * sigma_safe / scale
    
    # Combined score: weighted sum of PoI/cost, EI/scale, and exploration
    # PoI/cost gives strong incentive to find improvements cheaply
    # EI/scale adds exploitation of large expected gains
    # Exploration bonus ensures we don't get stuck in local optima
    scores = 0.6 * poi_cost + 0.35 * ei_scaled + exploration_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t_start
    if elapsed > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
