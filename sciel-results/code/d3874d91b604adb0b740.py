"""Acquisition function for Bayesian optimization of beale 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Quantile Acquisition Function (QAF) with adaptive exploration.
    
    Uses the probability of improvement as the core signal:
        P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    
    This directly optimizes for the chance of finding a new minimum.
    An adaptive exploration bonus is added to encourage visiting
    uncertain regions early in the search, decaying as f_best
    approaches the global minimum of the Beale function (0.0).
    
    Parameters
    ----------
    mu : array_like, shape (n,)
        Posterior mean at candidate points.
    sigma : array_like, shape (n,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed function value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against non-finite values
    mu = np.where(np.isfinite(mu), mu, np.nanmax(mu) if np.any(np.isfinite(mu)) else 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # Core QAF signal: probability that f(x) < f_best
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    prob_improve = norm.cdf(z)
    
    # Adaptive exploration bonus
    # When f_best is large (early), we want more exploration.
    # When f_best is small (late), we want more exploitation.
    # Use a smooth decay based on f_best.
    # Beale function range is roughly [0, 100+].
    # Let's use a sigmoid or similar to map f_best to an exploration weight.
    # weight = 1 / (1 + exp(k * (f_best - center)))
    # If f_best is large (e.g. 50), weight is small? No, we want LARGE exploration early.
    # So weight should be large when f_best is large.
    # Let's use: weight = 1 + alpha * (1 / (1 + exp(-k * (f_best - center))))
    # Or simply: weight = 1 + alpha * sigmoid(f_best)
    # If f_best = 0, sigmoid(0) = 0.5. weight = 1 + 0.5*alpha.
    # If f_best = 10, sigmoid(10) ~ 1. weight = 1 + alpha.
    # If f_best = 50, sigmoid(50) ~ 1. weight = 1 + alpha.
    
    # Let's try a linear-ish decay that saturates.
    # exploration_scale = 1.0 + 2.0 / (1.0 + f_best)
    # If f_best = 0, scale = 3.0
    # If f_best = 1, scale = 2.0
    # If f_best = 10, scale = 1.18
    # If f_best = 100, scale = 1.02
    # This encourages exploration when f_best is small?
    # Wait, if f_best is small, we are close to the optimum. We should EXPLOIT.
    # If f_best is large, we are far. We should EXPLORE.
    # So exploration_scale should be HIGH when f_best is HIGH.
    
    # Corrected logic:
    # exploration_scale = 1.0 + alpha * (1 - exp(-f_best / scale))
    # If f_best = 0, scale = 1.0
    # If f_best = 10, scale = 1 + alpha * (1 - exp(-10/scale)) ~ 1 + alpha
    # Let's use scale = 5.0, alpha = 2.0
    # f_best=0 -> 1.0
    # f_best=5 -> 1 + 2*(1 - exp(-1)) = 1 + 2*0.632 = 2.26
    # f_best=20 -> 1 + 2*(1 - exp(-4)) = 1 + 2*0.982 = 2.96
    # f_best=100 -> ~3.0
    
    f_best_safe = max(f_best, 0.0)
    scale = 5.0
    alpha = 2.0
    exploration_scale = 1.0 + alpha * (1.0 - np.exp(-f_best_safe / scale))
    
    # The exploration bonus should be proportional to sigma.
    # score = prob_improve + exploration_scale * sigma * normalization_factor?
    # prob_improve is in [0, 1].
    # sigma can be large.
    # We need to balance them.
    # Let's look at the magnitude of sigma.
    # In standardized targets, sigma is in objective units.
    # If f_best is 10, sigma might be 1-5.
    # prob_improve is ~0.5.
    # If we add 3 * sigma, and sigma=2, we add 6. That dominates prob_improve.
    # This might be good for exploration.
    # But in late stage, f_best=0.1, sigma=0.01.
    # exploration_scale = 1 + 2*(1-exp(-0.02)) ~ 1.04.
    # Bonus = 1.04 * 0.01 = 0.0104.
    # prob_improve is still dominant.
    
    # Let's try: scores = prob_improve + exploration_scale * sigma
    # But we need to be careful with units.
    # Maybe normalize sigma?
    # Or just use it as is. The relative ordering of sigma matters.
    
    scores = prob_improve + exploration_scale * sigma
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check for time (though this is fast)
    if time.time() - t_start > 300:
         # This should never happen for this simple calculation
         pass
        
    return scores
# EVOLVE-BLOCK-END
