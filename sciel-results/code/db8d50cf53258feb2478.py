"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI) acquisition function.
    # Favors points with a high likelihood of surpassing the current best.
    # Add a small jitter to sigma to avoid division by zero.
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    scores = norm.cdf(z)
    
    # Combine PI with a small exploration bonus based on uncertainty
    # to avoid getting stuck in local optima
    # Use a weighted combination: primarily PI, with a small UCB-like term
    # The exploration term encourages evaluating uncertain regions
    
    # Normalize scores to [0, 1] range for stable combination
    scores_min = np.min(scores)
    scores_max = np.max(scores)
    scores_range = scores_max - scores_min
    if scores_range > 1e-12:
        scores_normalized = (scores - scores_min) / scores_range
    else:
        scores_normalized = np.ones_like(scores) * 0.5
    
    # Exploration bonus: favor higher sigma (uncertainty)
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        exploration = sigma_safe / sigma_max
    else:
        exploration = np.zeros_like(sigma_safe)
    
    # Weight: 95% PI, 5% exploration
    weights = 0.95
    final_scores = weights * scores_normalized + (1 - weights) * exploration
    
    # Ensure finite values
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    return final_scores
# EVOLVE-BLOCK-END
