import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a hybrid of Probability of Improvement (PI) and normalized UCB,
    with adaptive cost weighting that shifts from exploration-heavy to
    exploitation-heavy as the budget is consumed.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    sigma = np.maximum(sigma, 1e-8)
    
    # Clip z for numerical stability
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    # Probability of Improvement: P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # Normalized UCB: balance exploitation and exploration
    # Use a time-varying beta that starts high (exploration) and decays
    # We estimate progress by looking at how good f_best is relative to sigma scale
    # Early on, f_best is likely far from optimum, so we want more exploration
    # Later, f_best is closer, so we want more exploitation
    
    sigma_max = np.max(sigma)
    if sigma_max > 1e-8:
        # Normalized uncertainty: high where model is uncertain
        norm_sigma = sigma / sigma_max
    else:
        norm_sigma = np.zeros_like(sigma)
    
    # Adaptive beta: starts at 2.0 (exploration-heavy) and decays to 0.5
    # Use a simple heuristic based on the spread of pi values
    # If pi values are widely spread, we're in exploration phase
    # If concentrated, we're in exploitation phase
    pi_spread = np.std(pi)
    # Map spread [0, 0.5] to beta [2.0, 0.5]
    # When spread is high (early), beta is high (more exploration)
    # When spread is low (late), beta is low (more exploitation)
    beta = 0.5 + 1.5 * (1.0 - np.clip(pi_spread / 0.5, 0, 1))
    
    # UCB score: exploitation (PI) + exploration (normalized uncertainty)
    ucb_score = pi + beta * norm_sigma
    
    # Cost-aware weighting: use a power law that's less aggressive than 1/cost
    # cost is in [1, 2], so cost^-0.7 maps to [1.0, ~0.616]
    # This smoothly penalizes expensive points without being too harsh
    cost_weight = np.power(cost, -0.7)
    
    scores = ucb_score * cost_weight
    
    # Deadline guard
    if time.time() - t_start > 2.0:
        scores = norm_sigma
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
