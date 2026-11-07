import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Safety: ensure finite inputs and avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # Compute Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware adjustment:
    # Combine a gentle multiplicative discount with a small additive penalty.
    # cost is in [1, 2], so (cost - 1) is in [0, 1].
    
    # Multiplicative factor: mildly downweights expensive points
    # factor ranges from 1.0 (cost=1) to ~0.909 (cost=2)
    alpha_mult = 0.1
    cost_factor = 1.0 / (1.0 + alpha_mult * (cost - 1.0))
    
    # Additive penalty: small linear penalty in cost
    # lambda_add * cost ranges from ~0.1 to ~0.2
    lambda_add = 0.1
    cost_penalty = lambda_add * cost
    
    # Combine: apply multiplicative discount first, then subtract additive penalty
    scores = ei * cost_factor - cost_penalty
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
