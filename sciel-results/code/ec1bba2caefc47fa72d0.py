"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Hybrid Expected Improvement (EI) and Upper Confidence Bound (UCB) acquisition
    function for minimization.
    
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    UCB_term = beta * sigma(x)
    
    Score = EI(x) + UCB_term
    
    This balances exploitation (via EI, which favors points with high potential
    improvement) and exploration (via UCB term, which favors points with high
    uncertainty). The EI term dominates when the model is confident about
    improvement, while the UCB term encourages visiting unexplored regions
    where the GP variance is high.
    
    Parameters:
        mu: (2000,) array of GP means in objective units.
        sigma: (2000,) array of GP standard deviations in objective units.
        f_best: Lowest observed function value.
    
    Returns:
        (2000,) array of scores, higher meaning more desirable.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # E[max(f_best - f(x), 0)]
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration term: UCB-style bonus for high uncertainty
    # We add a fraction of the standard deviation to encourage exploration.
    # A coefficient of 0.5 is a reasonable starting point for Branin-like functions.
    exploration_weight = 0.5
    ucb_bonus = exploration_weight * sigma_safe
    
    # Combined score
    scores = ei + ucb_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
