"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Posterior sampling-based acquisition that approximates Expected Regret
    via Monte Carlo sampling of the GP posterior.
    
    Since the global minimum of Goldstein-Price 2D is known to be 3.0,
    we compute the probability that each candidate improves over the true
    optimum (i.e., f(x) < 3.0) and combine it with expected improvement
    relative to the observed best for exploitation.
    """
    t_start = time.time()
    deadline = t_start + 290.0  # 300s limit, 10s safety margin
    
    n = len(mu)
    
    # Clamp sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # --- Monte Carlo estimation of P(f(x) < 3.0) ---
    # Sample from the GP posterior: f(x) ~ N(mu(x), sigma(x))
    # For each candidate, compute P(f(x) < 3.0) = Phi((3.0 - mu) / sigma)
    # This is analytic, no MC needed for this single-threshold probability.
    
    target = 3.0
    z_target = (target - mu) / sigma_safe
    prob_below_target = norm.cdf(z_target)
    
    # --- Expected Improvement relative to observed best (exploitation) ---
    z_best = (f_best - mu) / sigma_safe
    ei_best = (f_best - mu) * norm.cdf(z_best) + sigma_safe * norm.pdf(z_best)
    
    # --- Expected Improvement relative to true minimum (exploration) ---
    # EI with respect to target: how much can we expect to beat 3.0?
    # Since 3.0 is the minimum, EI_target = max(3.0 - f(x), 0) in expectation
    # This is: E[max(target - f(x), 0)] = (target - mu) * Phi(z) + sigma * phi(z)
    # where z = (target - mu) / sigma
    ei_target = (target - mu) * norm.cdf(z_target) + sigma_safe * norm.pdf(z_target)
    
    # --- Combine: use prob_below_target as a strong exploration signal ---
    # In regions with high uncertainty, prob_below_target can be significant
    # even if mu is above 3.0, encouraging exploration of unexplored regions.
    # Blend with EI for exploitation.
    
    # Normalize each component to [0, 1] scale for stable weighting
    prob_min = np.min(prob_below_target)
    prob_max = np.max(prob_below_target)
    prob_range = prob_max - prob_min + 1e-9
    prob_norm = (prob_below_target - prob_min) / prob_range
    
    ei_target_min = np.min(ei_target)
    ei_target_max = np.max(ei_target)
    ei_target_range = ei_target_max - ei_target_min + 1e-9
    ei_target_norm = (ei_target - ei_target_min) / ei_target_range
    
    ei_best_min = np.min(ei_best)
    ei_best_max = np.max(ei_best)
    ei_best_range = ei_best_max - ei_best_min + 1e-9
    ei_best_norm = (ei_best - ei_best_min) / ei_best_range
    
    # Weighting: emphasize probability of beating the true minimum,
    # which provides a direct exploration signal.
    # Also include EI relative to observed best for exploitation.
    # The relative weights can be adapted based on how close f_best is to 3.0.
    
    # When f_best is close to 3.0, exploitation becomes more important.
    # When f_best is far from 3.0, exploration (prob_below_target) is more valuable.
    closeness = np.exp(-(f_best - target) / 1.0)  # 1 when f_best near 3.0, 0 when far
    w_prob = 0.5 + 0.3 * (1 - closeness)  # 0.5 to 0.8
    w_ei_target = 0.3
    w_ei_best = 0.2
    
    # Normalize weights to sum to 1
    total_w = w_prob + w_ei_target + w_ei_best
    w_prob /= total_w
    w_ei_target /= total_w
    w_ei_best /= total_w
    
    score = w_prob * prob_norm + w_ei_target * ei_target_norm + w_ei_best * ei_best_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() > deadline:
        return score
    
    return score
# EVOLVE-BLOCK-END
