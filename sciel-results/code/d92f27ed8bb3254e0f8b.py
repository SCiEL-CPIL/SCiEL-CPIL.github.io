import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Independent Thompson Sampling.
    """
    start_time = time.time()
    
    # Input validation and sanitization
    n = len(mu)
    if n == 0:
        return np.array([])
        
    # Handle non-finite values
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        # Fallback to cost-aware greedy
        safe_cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        safe_mu = np.where(np.isfinite(mu), mu, 0.0)
        return -safe_mu / safe_cost

    # Ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Independent Thompson Sampling
    # Sample a value from the posterior predictive distribution for each candidate
    # y_i ~ N(mu_i, sigma_i)
    # We use np.random.normal for speed and simplicity
    samples = np.random.normal(mu, sigma)
    
    # The acquisition score is the negative of the sampled value, divided by cost
    # Higher score means better (lower sampled value and/or lower cost)
    scores = -samples / cost
    
    # Deadline guard
    # The operation above is very fast (vectorized). 
    # If it were slow, we would check time here.
    elapsed = time.time() - start_time
    if elapsed > 290:
        # In a more complex implementation, we might return a partial result or fallback
        # Here, we just return the computed scores as they are likely valid
        pass
        
    return scores
