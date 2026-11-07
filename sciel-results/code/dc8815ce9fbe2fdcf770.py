"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_DEADLINE = 325.0  # safety margin before 330s kill
_start_time = time.time()


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Gaussian EI with adaptive exploration bonus.
    
    Uses EI plus a carefully tuned exploration term that balances exploitation
    and exploration for the 6D Hartmann function with limited budget (30 evals).
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        return sigma
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Standard Gaussian Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive exploration: use 0.15 * sigma for better coverage in 6D
    # with only 5 initial points. This is slightly more exploratory than 0.1
    # to help escape local minima in the early stages.
    exploration_bonus = 0.15 * sigma
    
    # Small penalty for very low mu values that are already better than f_best
    # This prevents the algorithm from getting stuck if it finds a good point
    # but still has budget to explore
    # Only apply if mu is significantly better than f_best
    improvement = np.maximum(f_best - mu, 0)
    # Soft penalty: 0.01 * improvement^2, capped at 0.05
    penalty = np.minimum(0.01 * improvement**2, 0.05)
    
    scores = ei + exploration_bonus - penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
