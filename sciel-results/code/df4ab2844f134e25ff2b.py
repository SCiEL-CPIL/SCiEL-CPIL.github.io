import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 15.0

def _time_exceeded():
    return (time.time() - _START_TIME) > (_TIME_LIMIT - _SAFETY_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian Optimization.
    Uses Cost-Aware Expected Improvement (cEI) with a hard feasibility gate.
    
    Parameters:
    mu (np.array): Mean predictions from the GP, shape (N,).
    sigma (np.array): Standard deviation predictions from the GP, shape (N,).
    f_best (float): The best function value observed so far.
    cost (np.array): Cost of evaluating each candidate point, shape (N,).
    
    Returns:
    np.array: Scores for each candidate, higher is better.
    """
    # Time guard: If we are running out of time, return a safe default
    # (exploit best mu with minimal cost preference) to avoid timeout.
    if _time_exceeded():
        # Simple exploitation: prefer low mu, break ties by low cost
        cost_safe = np.maximum(cost, 1e-9)
        scores = -mu / cost_safe
        return np.where(np.isfinite(scores), scores, -np.inf)
    
    # Ensure sigma is non-negative and handle numerical issues
    sigma = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF and phi is PDF of standard normal.
    
    # For points where sigma is very small, EI is approximately max(0, f_best - mu)
    # For points where sigma is large, EI has a different shape.
    
    # Compute z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Handle potential overflow/underflow in z
    z = np.clip(z, -10, 10)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: For minimization, improvement is f_best - f(x), so if mu < f_best, we expect improvement.
    # The standard EI formula for minimization with prior best f_best is:
    # EI(x) = E[max(f_best - f(x), 0)]
    # If f(x) ~ N(mu, sigma^2), then:
    # EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    c_ei = ei / cost_safe
    
    # Hard feasibility gate based on cost quantiles:
    # This forces the search to concentrate in low-cost regions.
    # Use 70th percentile as the threshold: only consider points with cost below this.
    cost_threshold = np.percentile(cost, 70)
    
    # Create a mask: points with cost <= threshold get their c_ei, others get a heavily penalized score
    # To make this a hard gate, we can set scores for expensive points to a very low value
    # relative to the best c_ei, or simply zero them out and let the best low-cost point win.
    
    # A softer but still strong gate: multiply by a factor that drops off sharply above the threshold.
    # Or simply: if cost > threshold, score = -inf (or very negative).
    # Let's use a hard gate: points above the 70th percentile of cost get a score of -1e6 * max(c_ei)
    # This effectively excludes them unless no low-cost points are available.
    
    max_cei = np.max(c_ei)
    if max_cei <= 0:
        # If all EI are zero, just pick the cheapest point
        scores = -cost_safe
    else:
        # Hard gate: penalize high-cost points heavily
        gate = np.where(cost <= cost_threshold, 1.0, 1e-6)
        scores = c_ei * gate
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
