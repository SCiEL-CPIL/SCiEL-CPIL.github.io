"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin of 10s before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Args:
        mu: (N,) array of GP mean predictions.
        sigma: (N,) array of GP standard deviations.
        f_best: float, best observed value so far.
        cost: (N,) array of costs for each candidate.
        
    Returns:
        (N,) array of acquisition scores (higher is better).
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a safe, neutral score if we are running out of time
        # This effectively stops the search or picks a random point if forced
        return np.zeros_like(mu, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    if mu.size == 0:
        return np.array([])
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Thompson Sampling:
    # 1. Sample a single value from the posterior distribution for each candidate.
    #    Since we are minimizing, we want low values.
    #    The posterior is Normal(mu, sigma).
    #    We sample z ~ N(0, 1).
    #    sample = mu + sigma * z
    
    # Use a fixed seed based on f_best to ensure some consistency but allow variation 
    # if f_best changes significantly? 
    # Actually, standard TS uses a random seed per call. 
    # However, in this batch context, we want the "best" sample among the 2000.
    # If we use random noise, the score is stochastic.
    # The evaluator picks the argmax of the returned scores.
    # So we return the sampled values. The evaluator will pick the one with the lowest sampled value?
    # Wait, the contract says "higher = evaluate next".
    # So we should return -sampled_value / cost.
    
    # Generate random noise
    # Using a fixed seed might make the behavior deterministic across calls if f_best doesn't change,
    # which could lead to re-evaluating the same point if the GP doesn't update enough.
    # But GP updates every time we evaluate a new point.
    # To ensure we don't get stuck in a local loop due to deterministic noise,
    # we can use a seed derived from the current state or just use a global random state.
    # Let's use a global random state to ensure true randomness.
    
    # Note: Using np.random.rand is fast.
    z = np.random.randn(mu.shape)
    
    # Sampled objective value (lower is better)
    sampled_val = mu + sigma_safe * z
    
    # Acquisition score: We want to maximize this.
    # Since we are minimizing the objective, we want to pick the candidate with the LOWEST sampled_val.
    # So we return -sampled_val.
    # Additionally, we account for cost.
    # Cost-aware TS: Maximize -sampled_val / cost.
    # This favors points that are both likely to be good (low sampled_val) and cheap (low cost).
    
    score = -sampled_val / cost_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # If all scores are -inf (shouldn't happen with finite inputs), return zeros
    if not np.any(np.isfinite(score)):
        return np.zeros_like(mu, dtype=np.float64)
        
    return score
# EVOLVE-BLOCK-END
