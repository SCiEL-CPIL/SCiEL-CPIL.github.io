"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280  # 300s limit, keep 20s safety margin


def _check_deadline():
    """Return True if we should stop computing and return safe result."""
    return (time.time() - _START_TIME) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a multi-component blend:
    - Expected Improvement (EI) for exploitation
    - Upper Confidence Bound (UCB) for exploration  
    - Probability of Improvement (PI) for stability
    All normalized by cost with adaptive exponent.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Convert to numpy arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Compute z-score for improvement
    z = (f_best - mu) / sigma
    
    # Probability of Improvement: P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB: explore regions with high uncertainty
    # Adaptive beta: start high for exploration, decrease as f_best improves
    # f_best ranges from ~10 (early) to ~0 (converged)
    # beta goes from ~3.0 early to ~0.5 late
    beta = np.clip(3.0 - 2.5 * (1.0 - np.clip(f_best / 10.0, 0, 1)), 0.5, 3.0)
    ucb = mu + beta * sigma
    
    # For minimization, lower UCB is better, so we negate for "higher = better" scoring
    # Actually, we want to prefer points where mu is low AND sigma is high
    # UCB for minimization: mu - beta*sigma (lower is better)
    # Convert to "higher = better" by negating: -mu + beta*sigma
    ucb_score = -mu + beta * sigma
    
    # Normalize each component to [0, 1] range for stable blending
    # EI normalization: divide by scale of objective
    ei_scale = np.std(mu) + 1e-9
    ei_norm = ei / (ei_scale * 5 + 1e-9)  # Scale EI to be in similar range as PI
    ei_norm = np.clip(ei_norm, 0, 1)
    
    # UCB normalization
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_norm = (ucb_score - ucb_min) / ucb_range
    ucb_norm = np.clip(ucb_norm, 0, 1)
    
    # PI is already in [0, 1]
    pi_norm = np.clip(pi, 0, 1)
    
    # Adaptive weights based on f_best (how close we are to optimum)
    # Early on (f_best large): more exploration (UCB) and exploitation (EI)
    # Late on (f_best small): more stability (PI) and careful exploitation
    progress = np.clip(1.0 - f_best / 10.0, 0, 1)  # 0 = early, 1 = late
    
    # Weight schedule:
    # Early: EI=0.4, UCB=0.4, PI=0.2
    # Late:  EI=0.3, UCB=0.2, PI=0.5
    w_ei = 0.4 - 0.1 * progress
    w_ucb = 0.4 - 0.2 * progress
    w_pi = 0.2 + 0.3 * progress
    
    # Normalize weights to sum to 1
    w_sum = w_ei + w_ucb + w_pi
    w_ei /= w_sum
    w_ucb /= w_sum
    w_pi /= w_sum
    
    # Combine components
    combined = w_ei * ei_norm + w_ucb * ucb_norm + w_pi * pi_norm
    
    # Cost normalization with adaptive exponent
    # Use cost^0.5 for moderate cost sensitivity
    # Scale cost to [1, 2] range since cost = 1 + mean(x_scaled) where x_scaled in [0,1]
    cost_normalized = (cost - np.min(cost)) / (np.max(cost) - np.min(cost) + 1e-9)
    cost_factor = 1.0 + cost_normalized  # Range [1, 2]
    
    scores = combined / (cost_factor ** 0.5)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
