import numpy as np
import time
from scipy.stats import norm

# Global deadline tracking
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Knowledge Gradient (KG) acquisition function with cost awareness.
    
    The Knowledge Gradient for a candidate x, given the current best f_best,
    is the expected reduction in the best observed value upon evaluating x.
    For a Gaussian posterior N(mu, sigma^2), this is analytically equivalent
    to Expected Improvement (EI) for minimization.
    
    We compute EI(x) / cost(x) to prioritize evaluations that offer the best
    expected improvement per unit cost.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviations.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Evaluation costs for each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        mu_arr = np.asarray(mu)
        # Fallback: pure exploitation
        fallback_scores = -mu_arr
        return np.where(np.isfinite(fallback_scores), fallback_scores, 0.0)

    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Safety clamps
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    f_best_val = float(f_best)
    
    # Calculate Expected Improvement (KG) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best_val - mu) / sigma_safe
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z = np.clip(z, -10.0, 10.0)
    
    ei_scores = (f_best_val - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI should be non-negative
    ei_scores = np.maximum(ei_scores, 0.0)
    
    # Normalize EI scores to [0, 1] range to make cost division more stable
    # This ensures that the relative importance of EI is preserved
    ei_max = np.max(ei_scores)
    if ei_max > 1e-8:
        ei_normalized = ei_scores / ei_max
    else:
        ei_normalized = ei_scores
    
    # Cost-awareness: Divide by cost
    # Using linear cost penalty (exponent 1.0)
    # This directly maximizes expected improvement per unit cost
    scores = ei_normalized / cost_safe
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
