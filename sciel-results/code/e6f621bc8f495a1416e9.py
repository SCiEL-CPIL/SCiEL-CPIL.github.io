"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for time management
_start_time = None

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def _check_deadline(limit=320.0):
    """Returns True if we are still within the safe time limit."""
    return (time.time() - _get_start_time()) < limit

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases where cost might be 0 or negative (though spec says 1 + mean(x) so >= 1)
    # But sigma can be 0.
    safe_sigma = np.maximum(sigma, 1e-8)
    safe_cost = np.maximum(cost, 1e-8)
    
    # Thompson Sampling: Sample from the posterior
    # Since we don't have a random seed passed in, we use global random state or numpy's global RNG
    # To ensure determinism across runs if the evaluator expects it, we might need to be careful, 
    # but TS is inherently stochastic. The task says "seeded instances" for the function, 
    # but the acquisition function itself doesn't receive a seed.
    # Standard practice in these loops: use a fixed random seed for the acquisition noise 
    # or rely on the global numpy seed set by the harness. 
    # Given "five seeded instances", the harness likely sets np.random.seed before calling.
    
    # Sample from N(mu, sigma^2)
    # We scale the noise by 1/sqrt(cost) to penalize high-cost points more heavily in the sample
    # This makes the effective "variance" lower for high-cost points, making them less likely to be selected
    # unless their mean is significantly better.
    
    # Alternative formulation: 
    # score = (mu - f_best) / cost + noise * sigma / cost
    # This is equivalent to sampling y ~ N(mu, sigma^2) and scoring (y - f_best) / cost
    
    # Let's use the direct sampling approach which is standard TS
    # y = mu + sigma * z, where z ~ N(0, 1)
    z = np.random.randn(len(mu))
    y = mu + safe_sigma * z
    
    # Score is the improvement over f_best, divided by cost
    # If y < f_best, improvement is positive.
    # We want to maximize this score.
    scores = (y - f_best) / safe_cost
    
    # Safety check: if any non-finite values appear, replace with -inf (or a very low number)
    # The spec says "Return 2000 finite scores". 
    # If sigma is huge, y could be inf. 
    # Let's clip scores to a reasonable range to ensure finiteness, or handle NaNs.
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
