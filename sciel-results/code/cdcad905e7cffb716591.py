import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running out of time, return a safe default
    if time.time() > _DEADLINE - 5.0:
        # Fallback to a simple deterministic score if time is critical
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -10, 10)
        from scipy.stats import norm
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0)
        return ei / cost

    # Ensure sigma is non-negative and not zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: Draw a single sample from the posterior predictive distribution
    # theta = mu + sigma * z, where z ~ N(0, 1)
    # Using a fixed seed for the random draw within the acquisition function is not possible 
    # as we don't have access to the instance seed, but we can use a global random state 
    # or simply draw from a standard normal distribution. 
    # Since the function is called repeatedly, using a fresh random draw each time 
    # introduces stochasticity in the search, which is the point of Thompson Sampling.
    
    # However, to ensure reproducibility or stability if required by the framework,
    # one might consider using a deterministic pseudo-random number based on indices,
    # but standard Thompson Sampling uses i.i.d. draws.
    # Let's use np.random.standard_normal for the draw.
    z = np.random.standard_normal(len(mu))
    
    # Sampled function values
    theta = mu + sigma * z
    
    # Normalized regret: (theta - f_best) / sqrt(cost)
    # We want to MINIMIZE this.
    # The acquisition function must return scores where HIGHER = evaluate next.
    # So we return the negative of the normalized regret.
    
    # Ensure cost is positive to avoid division by zero or negative costs
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate normalized regret
    regret = (theta - f_best) / np.sqrt(cost_safe)
    
    # Score is negative regret (higher is better)
    score = -regret
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
