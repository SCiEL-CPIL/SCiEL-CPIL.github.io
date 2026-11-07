# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    start_time = time.time()
    
    # Input validation and sanitization
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases for sigma (avoid negative or zero variance issues)
    # Sigma is standard deviation, so it must be non-negative.
    sigma = np.maximum(sigma, 1e-9)
    
    # Handle edge cases for cost (avoid division by zero)
    cost = np.maximum(cost, 1e-9)
    
    # Thompson Sampling:
    # Sample a single objective value for each candidate from the Gaussian posterior.
    # y ~ N(mu, sigma^2)
    # We want to minimize the objective, so we prefer smaller y.
    # The "score" for the acquisition function should be high for candidates 
    # that are likely to yield a lower value than f_best.
    # Score = (f_best - y) / cost
    
    # Generate standard normal random numbers
    # Using np.random.randn for efficiency
    u = np.random.randn(mu.shape)
    
    # Sample y
    y = mu + sigma * u
    
    # Calculate the potential improvement (positive if y < f_best)
    # We use this as the base for the score.
    # If y > f_best, the improvement is negative, which is fine; 
    # those candidates will have lower scores.
    improvement = f_best - y
    
    # Cost-aware score
    # Higher score means we should evaluate this point next.
    # We divide by cost to penalize expensive evaluations.
    score = improvement / cost
    
    # Ensure finite values
    # If score is not finite (e.g., inf), replace with 0 or a very small number.
    # Since mu, sigma, cost are sanitized, score should be finite unless 
    # f_best is inf or NaN. f_best is from observations, so it should be finite.
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety margin check
    elapsed = time.time() - start_time
    if elapsed > 280:
        # This should not happen as the operations are fast, but just in case
        # Return a neutral score
        return np.zeros_like(mu)
        
    return score

# EVOLVE-BLOCK-END
