"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Aware Thompson Sampling acquisition function.
    
    Strategy:
    Uses Thompson Sampling by drawing a single sample from the GP posterior 
    for all candidates. The score is the sampled value divided by the cost.
    This stochastic approach allows for better exploration in multimodal 
    functions like Hartmann 3D compared to deterministic EI, as it naturally 
    balances exploration and exploitation based on the posterior uncertainty.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Fallback: simple greedy exploitation if time is critical
        # Pick the point with the lowest mu, penalized by cost
        scores = -mu / cost
        return np.where(np.isfinite(scores), scores, -np.inf)

    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw a single sample from the GP posterior: y ~ N(mu, sigma^2)
    # Using a fixed seed is not appropriate here as we want stochasticity across calls,
    # but we can use the global numpy random state.
    # Note: In a loop, this will provide different samples each time.
    samples = mu + sigma_safe * np.random.standard_normal(len(mu))
    
    # Cost-aware score: Sampled value per unit cost
    # We want to minimize the objective, so higher score is better.
    # Since we are sampling the objective value (lower is better),
    # we should select the point with the LOWEST sampled value per cost?
    # Wait, Thompson Sampling for minimization:
    # We sample y from p(y|x). We want to find x that minimizes y.
    # So we should select the x with the minimum y.
    # However, the contract says "Return 2000 finite scores (higher = evaluate next)".
    # So we need to invert the logic or negate the value.
    # Score = - (sampled_value / cost)
    # This way, a lower sampled value (better objective) and lower cost yields a higher score.
    
    scores = -samples / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
