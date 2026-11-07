"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global timer for deadline guard
_start_time = time.time()
_DEADLINE = 320.0  # 10s margin from 330s

def _check_deadline():
    return (time.time() - _start_time) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if _check_deadline():
        # If we are running out of time, return a score that favors cheap points
        # to at least complete the current iteration quickly.
        return -cost

    # Thompson Sampling strategy
    # We want to maximize the score.
    # For minimization, we want low function values.
    # We sample from the posterior: f ~ N(mu, sigma^2)
    # We then subtract a penalty for the cost.
    
    # To ensure reproducibility within a single call if needed, we could use a fixed seed,
    # but standard TS uses random draws. Since the evaluator calls this function repeatedly,
    # the randomness is part of the algorithm's exploration.
    
    # Draw samples from the GP posterior
    # mu and sigma are arrays of shape (2000,)
    # We generate random numbers with the same shape
    # Using np.random.normal for efficiency
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Sample: z ~ N(0, 1)
    # f_sampled = mu + sigma * z
    z = np.random.normal(0, 1, size=mu.shape)
    f_sampled = mu + sigma_safe * z
    
    # Cost-aware penalty
    # The cost is defined as 1 + mean(x_scaled). 
    # We want to minimize f(x) + lambda * cost(x).
    # So we maximize -(f_sampled + lambda * cost).
    # 
    # Choosing lambda:
    # The cost range is [1, 2].
    # The function values for Styblinski-Tang are in the range [0, 400] (approx).
    # The global minimum is -78.33.
    # A penalty of 0.1 * cost adds a penalty between 0.1 and 0.2.
    # This is small compared to the function variation but enough to break ties
    # or slightly prefer cheaper regions when uncertainty is similar.
    # 
    # Alternatively, we can use a larger penalty to strongly prefer cheap points.
    # However, if the penalty is too large, we might ignore high-value regions.
    # Let's try lambda = 0.5.
    
    lambda_penalty = 0.5
    scores = -(f_sampled + lambda_penalty * cost)
    
    # Ensure finite scores
    # If any NaNs or Infs sneak in, replace them with a very low score (e.g., -1e6)
    # so they are never selected.
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    return scores
# EVOLVE-BLOCK-END
