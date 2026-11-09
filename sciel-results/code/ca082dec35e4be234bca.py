"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard
_start_time = time.time()
_DEADLINE = 320.0  # 300s limit + 20s safety margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PI).
    
    PI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    
    Unlike EI, which weighs improvement by magnitude, PI focuses purely on the 
    probability of finding a better point. In low-budget scenarios (30 evals), 
    this can be more effective as it aggressively targets regions with high 
    confidence of improvement rather than balancing exploration/exploitation 
    via expected gain.
    
    mu: (N,) posterior mean in objective units
    sigma: (N,) posterior std in objective units
    f_best: best observed value so far in objective units
    
    Returns: (N,) scores, higher is better.
    """
    global _start_time
    
    # Deadline guard: if we are close to the limit, return a safe default
    # This is unlikely to trigger for a simple vectorized operation, but 
    # ensures compliance with the 300s constraint.
    if time.time() > _DEADLINE:
        return np.zeros_like(mu, dtype=np.float64)
    
    # Ensure inputs are numpy arrays of float64 for stability
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    # Clip sigma to a small positive value to avoid division by zero
    # If sigma is 0, the posterior is certain. 
    # If mu < f_best, PI should be 1. If mu >= f_best, PI should be 0.
    # However, standard PI formula with sigma->0:
    # z = (f_best - mu) / sigma
    # If mu < f_best, z -> +inf, Phi(+inf) = 1.
    # If mu > f_best, z -> -inf, Phi(-inf) = 0.
    # If mu == f_best, z -> 0, Phi(0) = 0.5.
    # This behavior is generally desirable.
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate standard normal variable z
    z = (f_best - mu) / sigma_safe
    
    # Calculate Probability of Improvement
    # PI = Phi(z)
    pi = norm.cdf(z)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(pi), pi, 0.0)
    
    return score
# EVOLVE-BLOCK-END
