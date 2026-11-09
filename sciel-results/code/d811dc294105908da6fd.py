"""Cost-aware acquisition function for Bayesian optimization on rastrigin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_DEADLINE = 280.0
_start_time = time.time()


def _check_deadline():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    This function replaces the previous Lower Confidence Bound (LCB) with 
    Expected Improvement normalized by cost. This structural change shifts 
    from a linear uncertainty-weighted score to a non-linear expected utility 
    score, which is often more effective for multimodal landscapes like 
    Rastrigin when combined with cost-awareness.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # Z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    
    # Handle the case where sigma is very small to avoid numerical issues
    # If sigma is tiny, EI approx max(0, f_best - mu)
    
    Z = (f_best - mu) / sigma_safe
    
    # Clip Z to a reasonable range to avoid overflow in exp for phi
    # Phi and phi are stable, but large Z leads to Phi->1, phi->0
    # We can just compute directly.
    
    # Phi(Z) is the CDF of the standard normal
    Phi = norm.cdf(Z)
    
    # phi(Z) is the PDF of the standard normal
    phi = norm.pdf(Z)
    
    # EI formula
    EI = (f_best - mu) * Phi + sigma_safe * phi
    
    # EI is always non-negative.
    # If f_best - mu is negative (mu > f_best), Z is negative.
    # Phi(Z) is small, phi(Z) is small.
    # The term (f_best - mu) * Phi(Z) will be negative * small = negative small.
    # sigma * phi(Z) is positive small.
    # Mathematically EI >= 0.
    # Due to floating point, it might be slightly negative. Clip to 0.
    EI = np.maximum(EI, 0.0)
    
    # Normalize by cost
    # The directive suggests "Expected Regret (ER) normalized by cost".
    # In this context, EI is the expected reduction in objective (gain).
    # Maximizing Gain / Cost is equivalent to minimizing Cost / Gain.
    # This directly optimizes the expected reduction per unit cost.
    
    scores = EI / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
