import numpy as np
import time

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Approximate Thompson Sampling.
    
    Instead of calculating a deterministic acquisition score like EI or UCB,
    this method samples from the Gaussian posterior distribution for each candidate
    and selects the candidate with the best (lowest) sampled value relative to its cost.
    
    This provides a structural shift from "maximizing expected improvement" to
    "sampling from the belief distribution and exploiting the best sample", which
    often performs well in high-dimensional or multimodal landscapes by avoiding
    the local minima traps of greedy EI.
    """
    global _start_time, _call_count
    
    # Initialize time tracking
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we've used most of the 330s budget, return quickly
    elapsed = time.time() - _start_time
    if elapsed > 315:
        # Fallback to simple LCB for speed
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        # Use a simple heuristic if we are out of time
        score = -mu / np.maximum(cost, 1e-9)
        score = np.where(np.isfinite(score), score, 0.0)
        return score

    _call_count += 1
    
    # Ensure arrays are 1D and finite
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Generate random samples from the posterior
    # y ~ N(mu, sigma^2)
    # We use a single random draw for all points to maintain correlation structure implicitly?
    # No, standard TS samples independently for each point.
    # However, since we don't have the covariance matrix, we assume independence for the sampling step.
    # This is a valid approximation if the GP is well-fitted or if we treat them as independent uncertainties.
    
    # To ensure reproducibility within a single call (though the function is called repeatedly),
    # we just use np.random. The evaluator will call this function sequentially.
    # We do NOT seed the random generator inside the function to allow for stochastic exploration
    # across different calls, which is the point of TS.
    
    # Generate standard normal random numbers
    z_samples = np.random.randn(len(mu))
    
    # Compute sampled values
    sampled_values = mu + sigma * z_samples
    
    # Cost-aware selection:
    # We want to minimize the sampled value, but penalize high cost.
    # A common cost-aware TS formulation is to minimize sampled_value / cost.
    # Since we return scores where HIGHER is better, we negate the ratio.
    # score = - (sampled_value / cost)
    
    # To avoid division by zero or very small costs, we use a safe cost
    safe_cost = np.maximum(cost, 1e-9)
    
    scores = -sampled_values / safe_cost
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
