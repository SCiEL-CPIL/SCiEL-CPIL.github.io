"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_ACQ_START_TIME = None
_ACQ_DEADLINE = 300.0  # Wall-clock limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI and UCB with adaptive weighting.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions for 2000 candidates.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions for 2000 candidates.
    f_best : float
        Current best observed value (lowest).
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate).
    """
    global _ACQ_START_TIME
    if _ACQ_START_TIME is None:
        _ACQ_START_TIME = time.time()
    
    elapsed = time.time() - _ACQ_START_TIME
    if elapsed > _ACQ_DEADLINE - 10:
        # Safety fallback: return EI/cost quickly
        sigma_safe = np.maximum(sigma, 1e-10)
        cost_safe = np.maximum(cost, 1e-10)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of being a new minimum
    prob_new_min = norm.cdf(z)
    
    # UCB for minimization: we want low mu and high sigma
    # Adaptive beta: higher when uncertainty is high relative to best
    # Use a smooth schedule based on relative uncertainty
    max_sigma = np.max(sigma_safe)
    if max_sigma > 1e-10:
        # beta increases with overall uncertainty, capped to avoid over-exploration
        beta = 0.5 + 1.0 * np.tanh(max_sigma / (abs(f_best) + 1e-6))
    else:
        beta = 0.5
    
    ucb_score = f_best - mu + beta * sigma_safe
    
    # Normalize both EI and UCB by cost
    # Use a smoother cost normalization: divide by (1 + cost) to avoid extreme values
    cost_norm = 1.0 + cost_safe
    ei_cost = ei / cost_norm
    ucb_cost = ucb_score / cost_norm
    
    # Adaptive weighting based on relative uncertainty per candidate
    # When sigma is high for this candidate, favor exploration (UCB)
    # When sigma is low, favor exploitation (EI)
    if max_sigma > 1e-10:
        sigma_norm = sigma_safe / max_sigma
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Sharper transition: use sigmoid-like weighting
    # w_ucb close to 0 when sigma is low, close to 0.7 when sigma is high
    w_ucb = 0.7 / (1.0 + np.exp(-8.0 * (sigma_norm - 0.5)))
    w_ei = 1.0 - w_ucb
    
    # Combine EI and UCB
    primary = w_ei * ei_cost + w_ucb * ucb_cost
    
    # Secondary term: bonus for high probability of being new minimum
    # Scaled by confidence to avoid rewarding pure noise
    confidence = prob_new_min * (1.0 - sigma_norm * 0.3)
    exploration_bonus = 0.15 * confidence * sigma_safe / cost_norm
    
    # Final score
    scores = primary + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
