"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time to avoid timeout
_start_time = time.time()
TIME_LIMIT = 320.0  # Safety margin (limit is 330s)

def _time_exceeded():
    return (time.time() - _start_time) > TIME_LIMIT

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (levy 2d).
    
    Strategy:
    1. Calculate Expected Improvement (EI) for minimization.
    2. Add an exploration bonus proportional to sigma to encourage novelty (diversity).
    3. Normalize by cost to prefer cheaper evaluations.
    4. Use median cost for normalization to be robust to outliers.
    """
    if _time_exceeded():
        # Fallback to simple EI/cost to be fast
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    # Calculate EI (minimization)
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If sigma is 0, EI is 0 if mu >= f_best, else f_best - mu.
    # Using 1e-9 to avoid division by zero.
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploration Term
    # We want to explore high uncertainty regions.
    # The "diversity penalty" finding suggests penalizing low sigma.
    # Adding kappa * sigma effectively does this, but let's make it stronger.
    # kappa = 1.0 seems like a good balance between exploitation and exploration.
    kappa = 1.0
    exploration_bonus = kappa * sigma
    
    # Combined Acquisition Value
    acq_value = ei + exploration_bonus
    
    # Cost Normalization
    # Normalize by median cost to make the score scale-invariant to the cost distribution.
    # This ensures that "expensive" points (high cost) are penalized relative to the average.
    median_cost = np.median(cost)
    if median_cost > 0:
        normalized_cost = cost / median_cost
    else:
        normalized_cost = cost + 1e-9
        
    # Final Acquisition Score
    # Higher is better.
    acq = acq_value / (normalized_cost + 1e-9)
    
    # Ensure finite values
    acq = np.where(np.isfinite(acq), acq, 0.0)
    
    return acq
# EVOLVE-BLOCK-END
