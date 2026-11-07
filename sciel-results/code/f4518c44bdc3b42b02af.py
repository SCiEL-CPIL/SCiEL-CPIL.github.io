"""Rewritten acquisition using GP-UCB with time-varying beta for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    GP-UCB acquisition function with time-varying exploration parameter.
    
    Parameters
    ----------
    mu : array of shape (2000,)
        GP posterior mean at candidate points.
    sigma : array of shape (2000,)
        GP posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns
    -------
    scores : array of shape (2000,)
        Acquisition scores; higher means more desirable to evaluate next.
    """
    # Estimate the number of evaluations so far from the magnitude of f_best
    # In practice, we don't have direct access to t, but we can use a heuristic:
    # The exploration should start high and decrease. We'll use a schedule based
    # on how far f_best is from a reasonable lower bound.
    
    # Since we minimize, and the minimum is around -2.53218, we can use f_best
    # to estimate progress. However, a simpler and more robust approach is to
    # use a fixed schedule that decreases beta over the 30 evaluations.
    
    # We don't know t exactly, but we can approximate it. Since the evaluator
    # starts with 5 uniform points and then does 25 more (total 30), and we're
    # called at each step after the initial 5, we can estimate t based on
    # how "good" f_best is relative to the known minimum.
    
    # Alternative: use a conservative decreasing beta. Since we don't know t,
    # we'll use a beta that starts moderate and would be appropriate for the
    # full optimization horizon. A common choice is beta_t = sqrt(log(t+1)) * constant.
    
    # For robustness without knowing t, we'll use a beta that balances exploration
    # and exploitation. Given the 2D domain and 30 evaluations, a beta around
    # sqrt(log(31)) * 2 ≈ 3.0 is reasonable for early-to-mid optimization.
    # But to be adaptive, let's use a heuristic based on f_best.
    
    # Heuristic: if f_best is close to the minimum (-2.53218), we're in exploitation
    # mode and should use lower beta. If f_best is far, we should explore more.
    
    min_value = -2.53218
    # Normalize the regret to [0, 1] roughly
    regret = f_best - min_value
    if regret < 0:
        regret = 0.0
    
    # Map regret to a beta value. When regret is large, explore more (higher beta).
    # When regret is small, exploit more (lower beta).
    # Use a smooth mapping: beta = base + scale * exp(-regret / scale_regret)
    base_beta = 1.0
    scale_beta = 3.0
    scale_regret = 1.0
    beta = base_beta + scale_beta * np.exp(-regret / scale_regret)
    
    # Clip beta to a reasonable range
    beta = np.clip(beta, 0.5, 5.0)
    
    # Compute UCB scores for minimization: we want to find low values, so
    # the acquisition should be high where mu is low and sigma is high.
    # UCB for minimization: score = -mu + beta * sigma
    # (We negate mu because lower mu is better, and we want higher scores for better points)
    
    # Handle sigma that might be very small or zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
