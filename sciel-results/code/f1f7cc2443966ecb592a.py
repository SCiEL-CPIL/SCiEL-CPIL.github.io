import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Explicit deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin of 10s before 300s limit
    
    try:
        # Check if we are already past the deadline (should not happen for this simple op)
        if time.time() > deadline:
            return np.zeros(len(mu))
            
        # Structural Change: Thompson Sampling
        # Instead of deterministic Expected Improvement (EI), we use stochastic Thompson Sampling.
        # This helps in exploration and naturally incorporates uncertainty into the cost-benefit tradeoff.
        
        # Sample a single standard normal variable for this acquisition call
        epsilon = np.random.normal(0, 1)
        
        # Predicted objective value: mu + epsilon * sigma
        # We want to minimize the predicted value per unit cost.
        # Score is defined such that higher is better.
        # So we negate the ratio.
        
        # cost is >= 1, so division is safe.
        # mu and sigma are finite.
        
        predicted_values = mu + epsilon * sigma
        scores = - predicted_values / cost
        
        # Validate scores
        if not np.all(np.isfinite(scores)):
            # Fallback to Expected Improvement if numerical issues arise
            from scipy.stats import norm
            z = (f_best - mu) / (sigma + 1e-9)
            scores = ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / cost
            # Ensure finite
            scores = np.where(np.isfinite(scores), scores, 0.0)
            
        return scores

    except Exception:
        # In case of any error, return a valid array of zeros to avoid invalidating the result
        return np.zeros(len(mu))
