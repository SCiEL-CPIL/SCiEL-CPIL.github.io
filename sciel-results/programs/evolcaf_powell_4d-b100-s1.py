"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: kill after 320s to stay safely under 330s
_DEADLINE = time.time() + 320.0


def _check_deadline():
    return time.time() < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Hybrid UCB + EI acquisition with adaptive weighting based on GP uncertainty.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - lowest observed value
        cost: array of shape (2000,) - evaluation costs
    
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    if not _check_deadline():
        # Fallback: return simple regret-based scores
        return (f_best - mu) / (cost + 1e-9)
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB (Upper Confidence Bound) - we're minimizing, so we want low mu + beta*sigma
    # Adaptive beta based on local uncertainty
    sigma_mean = np.mean(sigma_safe)
    sigma_std = np.std(sigma_safe) + 1e-9
    # Beta increases with overall uncertainty level
    beta = 1.0 + 2.0 * (sigma_mean / (sigma_std + 1e-9))
    beta = np.clip(beta, 0.5, 5.0)
    
    ucb = mu + beta * sigma_safe  # We want to minimize, so high UCB means uncertain
    
    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei) + 1e-9
    ucb_max = np.max(ucb) + 1e-9
    
    # Adaptive weighting: when sigma is high, favor UCB (exploration); when low, favor EI (exploitation)
    # Weight based on relative sigma
    sigma_rel = sigma_safe / (sigma_mean + 1e-9)
    w_ucb = np.clip(0.5 * sigma_rel, 0.1, 0.9)
    w_ei = 1.0 - w_ucb
    
    # Combined score: higher is better
    # EI is higher when improvement is expected; UCB is lower when we're confident
    # We want to maximize: w_ei * EI + w_ucb * (-UCB) normalized
    ei_norm = ei / ei_max
    ucb_norm = -ucb / (np.max(np.abs(ucb)) + 1e-9)  # Negate so higher = more promising
    
    combined = w_ei * ei_norm + w_ucb * (0.5 + 0.5 * ucb_norm)
    
    # Cost normalization: divide by cost to prefer cheaper evaluations
    scores = combined / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
