"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline tracking
_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: Probability of Improvement per Unit Cost.
    
    Strategy:
    Instead of Expected Improvement (EI), which can be dominated by high-variance
    regions with large potential gains but low confidence, we use the Probability
    of Improvement (PoI). We maximize the probability that f(x) < f_best - epsilon.
    This focuses the search on high-confidence regions where a significant improvement
    is likely, rather than just any improvement.
    
    We normalize by cost to respect the budget.
    """
    # Check deadline to ensure we return valid scores quickly
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)

    # Handle edge cases where sigma is very small (high confidence)
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Define a threshold epsilon for "significant" improvement.
    # Shekel 4D range is roughly [-10.5, 10]. 
    # An epsilon of 0.1 represents a meaningful step.
    # If f_best is very close to the minimum (-10.5364), we might want smaller eps.
    # However, a fixed epsilon is a standard PoI variant.
    # Let's use a small epsilon to encourage steady progress.
    epsilon = 0.05
    
    # Target value we want to beat
    target = f_best - epsilon
    
    # Calculate z-score: P(f(x) < target) = Phi((target - mu) / sigma)
    # Note: For minimization, we want f(x) < target.
    # If mu is much lower than target, z is large positive, Phi(z) -> 1.
    # If mu is much higher than target, z is large negative, Phi(z) -> 0.
    z = (target - mu) / sigma_safe
    
    # Probability of Improvement
    poi = norm.cdf(z)
    
    # Cost-aware normalization
    # Cost is in [1, 2]. 
    # We divide by cost to penalize expensive evaluations.
    # To avoid extreme values if cost is small, we add a small epsilon to cost.
    score = poi / (cost + 1e-6)
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
