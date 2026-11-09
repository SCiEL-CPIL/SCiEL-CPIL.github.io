"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware LCB-based acquisition function with exploration bonus.
    
    Structural idea: LCB-based scoring (confidence-based) rather than EI/KG (improvement-based).
    Refinement machinery: 
      - Probability-weighted exploitation term
      - Exploration bonus for high-uncertainty regions
      - Soft cost penalty (sqrt instead of linear)
    
    For minimization:
      - LCB = mu - beta * sigma (lower is better)
      - We want to maximize a score that favors low LCB and low cost
    """
    # Deadline guard
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
        
    elapsed = time.time() - acquisition._start_time
    if elapsed > 290:
        # Return a very cheap, fast computation
        return 1.0 / cost
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # LCB for minimization: lower is better
    beta = 1.5
    lcb = mu - beta * sigma_safe
    
    # Cost-aware adjustment: use sqrt(cost) for softer penalty
    cost_penalty = np.sqrt(cost)
    
    # Probability of improvement: P(f(x) < f_best)
    z = (f_best - mu) / sigma_safe
    prob_improve = norm.cdf(z)
    
    # Expected improvement
    expected_improvement = (f_best - mu) * prob_improve
    
    # Exploration bonus: probability of finding a significantly better point
    delta = 0.01
    z_delta = (f_best - delta - mu) / sigma_safe
    prob_significant = norm.cdf(z_delta)
    
    # Combined score: exploitation + exploration, all divided by cost penalty
    # Exploitation: expected improvement
    # Exploration: probability of significant improvement weighted by uncertainty
    score = (expected_improvement + 0.5 * prob_significant * sigma_safe) / cost_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
