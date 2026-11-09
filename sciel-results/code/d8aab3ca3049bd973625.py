"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed function value so far
    cost : array of shape (2000,) - cost of evaluating each point
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Time guard for safety, though this function is vectorized and fast
    start_time = time.time()
    deadline = start_time + 2.0 # 2 second soft limit for this function call
    
    # Handle edge cases where sigma might be very small or zero
    # We use a small epsilon to avoid division by zero in z-score
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-score for Expected Improvement (minimization)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (for minimization)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring
    # The cost is C(x) = 1 + mean(x_scaled). Range [1, 2].
    # We use a strict cost penalty: Score = EI / cost.
    # This prioritizes points with high expected improvement per unit cost.
    # In the low-budget regime (30 total), efficiency is critical.
    # The global minimum is at cost 1.5. Corners are cost 1.0 or 2.0.
    # By using alpha=1.0, we ensure that a point with 50% lower EI but 10% lower cost
    # is not preferred, but a point with 50% lower EI and 50% higher cost is penalized.
    
    # Check time
    if time.time() > deadline:
         # Fallback: return EI only (ignoring cost) if we are running out of time
         # This is unlikely but ensures we return finite scores
         scores = ei
         scores = np.where(np.isfinite(scores), scores, 0.0)
         scores = np.maximum(scores, 0.0)
         return scores

    # Apply cost penalty
    # Using alpha = 1.0 for strict cost-efficiency
    # This structural change removes the previous "exploration bonus" which was
    # suspected to waste budget on low-improvement, low-cost points.
    alpha = 1.0
    cost_factor = np.power(cost, alpha)
    
    scores = ei / cost_factor
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative scores
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
