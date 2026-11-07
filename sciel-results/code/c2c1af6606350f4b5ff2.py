"""Direct cost-aware acquisition for Bayesian optimization of rosenbrock 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Cost-Effective Expected Improvement (CEI).
    
    Strategy: Compute Expected Improvement (EI) over f_best, then divide by cost.
    This creates a non-linear, principled trade-off that more aggressively exploits
    cheap, promising points.
    
    EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    CEI = EI / cost
    
    We guard against the 330s wall-clock limit with an explicit deadline.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle NaN/Inf
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-Effective EI: EI / cost
    safe_cost = np.maximum(cost, 1e-8)
    score = ei / safe_cost
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 300:  # Additional safety check
        # Fallback: pure exploitation with cost awareness
        score = np.maximum(f_best - mu, 0.0) / safe_cost
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
