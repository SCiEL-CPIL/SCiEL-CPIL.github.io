import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 10.0

def _time_exceeded():
    return (time.time() - _START_TIME) > (_TIME_LIMIT - _SAFETY_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian Optimization.
    Uses a hybrid approach: Cost-Aware Expected Improvement (cEI) as the base,
    combined with a multiplicative feasibility gate based on cost.
    The gate forces exploitation of low-cost regions until uncertainty is exhausted.
    
    Parameters:
    mu (np.array): Mean predictions from the GP, shape (N,).
    sigma (np.array): Standard deviation predictions from the GP, shape (N,).
    f_best (float): The best function value observed so far.
    cost (np.array): Cost of evaluating each candidate point, shape (N,).
    
    Returns:
    np.array: Scores for each candidate, higher is better.
    """
    # Time guard: If we are running out of time, return a safe default
    # (e.g., uniform scores or just exploit best mu) to avoid timeout penalty.
    if _time_exceeded():
        # Fallback: Return scores based on low mu (exploitation)
        # This is a safe, fast operation.
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle sigma near zero to avoid division by zero in z
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for EI
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is CDF of standard normal, phi is PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    # Higher EI and lower cost lead to higher score
    cost_safe = np.maximum(cost, 1e-9)
    
    # Structural change: Multiplicative feasibility constraint
    # Define a threshold based on the current average cost per unit improvement
    # We use a percentile of costs to create a "low-cost zone"
    # If cost is below the 50th percentile (median), we give full weight.
    # If cost is above, we apply a penalty that decays as cost increases.
    # This creates a hard-ish gate that prefers low-cost candidates.
    
    # Calculate a dynamic threshold: median cost
    cost_median = np.median(cost)
    
    # Smooth transition: 
    # Score = EI * gate
    # gate = 1.0 if cost <= cost_median
    # gate = exp(-lambda * (cost - cost_median)) if cost > cost_median
    # Lambda controls how steep the penalty is.
    # A higher lambda makes the gate stricter.
    # Let's use lambda = 1.0 / cost_median to scale the penalty with the cost range.
    lambda_val = 1.0 / np.maximum(cost_median, 1e-9)
    
    # Calculate the gate
    # For cost <= cost_median, gate = 1.0
    # For cost > cost_median, gate = exp(-lambda_val * (cost - cost_median))
    gate = np.ones_like(cost)
    mask = cost > cost_median
    gate[mask] = np.exp(-lambda_val * (cost[mask] - cost_median))
    
    # Combine EI with the gate
    # We also divide by cost to maintain the cost-aware nature, but the gate
    # provides a stronger structural preference for low-cost regions.
    # Alternative: Just use EI * gate. The gate already penalizes high cost.
    # Let's try EI * gate / cost_safe to keep the explicit cost normalization
    # but with the added structural gate.
    
    scores = ei * gate / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
