"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for 2D Rosenbrock on [-2,2]^2.
    
    Strategy: Probability of improvement (POI) per unit cost, with a
    small expected-improvement component for tie-breaking. This is robust
    in the narrow valley of Rosenbrock where large improvements are rare
    but small steps are frequent.
    
    Parameters:
        mu: shape (2000,) - GP mean in objective units
        sigma: shape (2000,) - GP std in objective units
        f_best: float - lowest observed value
        cost: shape (2000,) - evaluation cost for each candidate
    
    Returns:
        shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're taking too long, return a simple fallback
    t0 = time.time()
    
    # Clip sigma to avoid division issues
    sigma = np.maximum(sigma, 1e-9)
    
    # z-score for improvement
    z = (f_best - mu) / sigma
    
    # Probability of improvement: P(f(x) < f_best) = Phi(z)
    poi = norm.cdf(z)
    
    # Expected improvement (small weight for tie-breaking among high-POI points)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize EI to [0, 1] scale relative to typical EI values
    # Use a soft normalization: divide by (f_best + typical scale)
    # Actually, just use a small weight since POI is the primary signal
    ei_weight = 0.1
    combined = poi + ei_weight * ei / (np.abs(f_best) + 1e-6)
    
    # Divide by cost for cost-awareness
    # cost = 1 + mean(x_scaled), so cost in [1, 2]
    score = combined / np.maximum(cost, 1e-6)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
