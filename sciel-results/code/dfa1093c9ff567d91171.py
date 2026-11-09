import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle zero sigma and cost to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Structural Change: Deterministic Cost-Adjusted Lower Confidence Bound (LCB)
    # We want to minimize f(x). 
    # The LCB is mu(x) - k * sigma(x).
    # We select the point that minimizes this value, adjusted for cost.
    # Acquisition score (higher is better) = - (mu - k * sigma) / cost
    # This is equivalent to (k * sigma - mu) / cost.
    
    # k controls the exploration-exploitation trade-off.
    # A higher k encourages exploration of areas with high uncertainty.
    # A lower k encourages exploitation of areas with low predicted values.
    # Given the tight budget, a moderate k is preferred.
    k = 2.5
    
    # Calculate the cost-aware LCB score
    # Note: We do NOT clip to 0 here because even if the expected improvement is negative,
    # we still want to pick the "least bad" candidate if all are bad, or the most promising 
    # unexplored candidate. The relative ordering is what matters for argmax.
    
    lcb_term = k * sigma_safe - mu
    scores = lcb_term / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Deadline guard
    if time.time() - start_time > 325:
        # Return a safe default if we are running out of time (unlikely for O(N) op)
        return np.zeros_like(mu)
        
    return scores
