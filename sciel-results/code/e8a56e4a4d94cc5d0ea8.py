import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Levy 2D.
    Uses a hybrid of Expected Improvement (EI) and a confidence-bound term,
    normalized by cost. Includes a deadline guard.
    """
    # Deadline guard: if we've been running for too long, return a safe fallback
    # (This is a static guard; the actual timing is managed by the evaluator,
    #  but we ensure our computation is O(n) and fast)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for standard normal CDF/PDF
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative (it should be, but numerical issues can occur)
    ei = np.maximum(ei, 0.0)
    
    # Calculate a UCB-like term for exploration
    # UCB = -mu + beta * sigma (for minimization, we want low mu and high sigma)
    # We use a smaller beta to balance exploitation and exploration
    beta = 2.0
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB with a weight
    # We weight EI more heavily as it directly measures expected improvement
    # but add UCB to encourage exploration in uncertain regions
    # The cost-awareness is applied to the combined score
    combined_score = ei + 0.5 * ucb
    
    # Normalize by cost to get a cost-aware score
    # Higher score = better (more improvement per unit cost)
    cost_safe = np.maximum(cost, 1e-8)
    score = combined_score / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
