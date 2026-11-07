"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: scalar - best observed value so far
    - cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    - scores: array of shape (2000,) - higher scores are preferred
    """
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, keep 10s safety margin
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Dynamic kappa for LCB: shifts from exploration to exploitation
    # Estimate remaining budget fraction based on typical costs
    mean_cost = np.mean(cost_safe)
    # Assume total budget is ~30, initial cost is ~5 (5 uniform points)
    # We don't know exact current total cost, but f_best and cost array give hints
    # Use a heuristic: as f_best approaches 0, we should exploit more
    # Also, higher sigma regions are more exploratory
    
    # Strategy: Use a weighted combination of EI/cost and LCB/cost
    # LCB: mu - kappa * sigma
    # Dynamic kappa: starts high (exploration), decreases as we find better solutions
    
    # Heuristic for kappa based on f_best relative to expected minimum
    # Griewank minimum is 0, so if f_best is close to 0, exploit
    # If f_best is still high, explore more
    
    # Scale f_best: assume typical range [0, 1] for griewank on this domain
    # But griewank can have values > 1, so use a robust scale
    f_scale = max(abs(f_best), 0.01)
    
    # kappa starts at ~2.0 and decreases to ~0.5 as f_best improves
    # Use exponential decay
    kappa = 0.5 + 1.5 * np.exp(-5.0 * f_best / f_scale)
    
    # LCB score
    lcb = mu - kappa * sigma_safe
    
    # Cost-normalized scores
    ei_score = ei / cost_safe
    lcb_score = -lcb / cost_safe  # Negate because lower lcb is better
    
    # Combine: weight EI more when uncertainty is high, LCB when we have good info
    # Use sigma as a proxy for uncertainty
    sigma_norm = sigma_safe / (np.max(sigma_safe) + 1e-10)
    
    # Weight: more EI when sigma is high (early), more LCB when sigma is low (later)
    w_ei = 0.5 + 0.5 * np.mean(sigma_norm)
    w_lcb = 1.0 - w_ei
    
    scores = w_ei * ei_score + w_lcb * lcb_score
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
