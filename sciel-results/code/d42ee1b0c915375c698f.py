"""Cost-aware acquisition using Cost-Effective Expected Improvement (CEI)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Module-level persistent state across calls within one evaluation run
_STATE = {
    "start_time": None,
}


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Effective Expected Improvement (CEI) acquisition function.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    # Deadline guard
    if _STATE["start_time"] is None:
        _STATE["start_time"] = time.time()
    
    elapsed = time.time() - _STATE["start_time"]
    if elapsed > 300.0:
        # Return a safe default to avoid timeout
        # Fallback to simple cost-aware LCB
        scores = (-mu - 2.0 * sigma) / np.maximum(cost, 1e-6)
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Handle sigma close to 0
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate Z-score for Expected Improvement (Minimization)
    # We want to maximize f_best - y
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If f_best - mu is negative, Phi(z) is small and phi(z) is small, 
    # but the first term is negative. 
    # Actually, EI is defined as E[max(0, f_best - y)].
    # If y > f_best, max is 0.
    # The formula (f_best - mu)Phi(z) + sigma phi(z) is always non-negative.
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: CEI = EI / cost
    cost_safe = np.maximum(cost, eps)
    scores = ei / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
