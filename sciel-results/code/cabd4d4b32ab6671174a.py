import numpy as np
from scipy.stats import norm
import time

# Global start time to track elapsed time for the entire evaluation session
_start_time = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s

def acquisition(mu, sigma, f_best):
    """
    Computes a hybrid acquisition function combining Expected Improvement (EI)
    and Upper Confidence Bound (UCB) for minimization.
    
    Strategy:
    - Uses a direct linear combination of EI and UCB without normalization.
    - EI provides exploitation pressure towards known good regions.
    - UCB provides exploration pressure in high-uncertainty regions.
    - This avoids the pitfalls of min-max normalization which can distort
      the acquisition landscape for uniform random candidate sets.
    
    Parameters:
    mu: array of shape (n_candidates,) - posterior mean
    sigma: array of shape (n_candidates,) - posterior standard deviation
    f_best: float - best observed value so far
    
    Returns:
    scores: array of shape (n_candidates,) - hybrid scores (higher is better)
    """
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Fallback: return a simple UCB to avoid hanging or complex computation
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 1.0 * sigma_safe

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Calculate Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid extreme values causing numerical issues in pdf/cdf
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    # EI should be non-negative for valid points, but clip just in case of numerical noise
    ei = np.maximum(ei, 0.0)
    
    # 2. Calculate Upper Confidence Bound (UCB) for minimization
    # UCB = -mu + beta * sigma
    # We use beta = 1.5 to encourage exploration in the multimodal Branin function.
    beta = 1.5
    ucb = -mu + beta * sigma_safe
    
    # 3. Combine EI and UCB
    # We use a weighted sum. Since both are in objective units, we can combine them directly.
    # alpha controls the relative weight of exploitation (EI) vs exploration (UCB).
    # A value of 0.5 gives equal weight to the magnitude of expected improvement
    # and the uncertainty-adjusted mean.
    alpha = 0.5
    scores = ei + alpha * ucb
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
