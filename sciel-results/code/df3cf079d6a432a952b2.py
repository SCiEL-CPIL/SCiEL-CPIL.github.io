"""Rewritten acquisition for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_ACQ_START_TIME = None
_DEADLINE_MARGIN = 5.0  # seconds before hard limit

def _check_deadline():
    """Return True if we should stop to respect the deadline."""
    global _ACQ_START_TIME
    if _ACQ_START_TIME is None:
        _ACQ_START_TIME = time.time()
    # The evaluator kills after 330s wall-clock. We don't know the total budget
    # per call, but we should be fast. Let's just check if elapsed since module load
    # is getting close to a reasonable limit. Actually, the contract says "Each evaluation
    # must finish within 300 s". But the function is called 25 times (30-5 initial).
    # So we have ~12s per call on average. Let's be very fast.
    # A simple check: if this call takes too long, we can't really do much inside
    # a vectorized numpy operation. So we just ensure our code is fast.
    # We'll use a global timer just in case.
    if time.time() - _ACQ_START_TIME > 295.0:  # Safety margin
        return True
    return False

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and a
    Knowledge Gradient (KG) bonus to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean predictions
    sigma : ndarray, shape (2000,)
        Posterior standard deviation predictions
    f_best : float
        Best observed value so far
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases where sigma is very small
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Knowledge Gradient bonus: rewards points with high probability of being new best
    # P(new best) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    p_new_best = norm.cdf(z)
    
    # Combine EI with a KG-inspired bonus
    # The bonus term 0.1 * sigma * P(new_best) encourages exploration
    # in uncertain regions that are likely to improve upon f_best
    kg_bonus = 0.1 * sigma_safe * p_new_best
    
    scores = ei + kg_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
