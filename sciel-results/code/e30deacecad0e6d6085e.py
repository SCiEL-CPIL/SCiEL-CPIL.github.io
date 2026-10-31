"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.special

_START_TIME = time.time()
_DEADLINE = 135.0  # safety margin before 150s kill

# Weibull distribution parameters for the generated sequences
# The task mentions "five generated Weibull sequences" with capacity 100.
# A common setup for these tasks uses Weibull(k, lambda) where items fit into bins of size 100.
# Assuming a standard Weibull with shape k=2.0 and scale lambda=25.0
# This yields a mean of ~22 and median of ~19, which is reasonable for capacity 100.
# We calculate the expected size of the next item.
# For Weibull(k, lambda), E[X] = lambda * Gamma(1 + 1/k)
_k = 2.0
_lambda_scale = 25.0
_EXPECTED_NEXT_ITEM = _lambda_scale * scipy.special.gamma(1 + 1/_k)

CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard: if we're too close to the limit, return a safe default
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        # Fallback: Best Fit (minimize remaining)
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores

    # Determine which bins have enough remaining capacity
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
        
    # Strategy: Distribution-Aware Best Fit with Dead Space Minimization
    #
    # We combine two structural insights:
    # 1. Best Fit Core: Prefer bins that leave the smallest remainder (consolidation).
    # 2. Distribution-Aware Dead Space Penalty: 
    #    Instead of a generic polarization penalty, we penalize remainders `r` 
    #    based on the "Dead Space" metric derived from the Weibull distribution.
    #    
    #    Dead Space Metric: D(r) = r * S(r) = r * exp(-(r/lambda)^k)
    #    This metric peaks at a specific 'dead zone' remainder where the space is 
    #    too small to likely fit a future item, but too large to be considered 
    #    negligible waste. 
    #    
    #    By subtracting this term from the score (making it more negative for dead 
    #    zones), we discourage creating remainders in the high-risk middle range, 
    #    pushing the algorithm towards either near-perfect fits (r~0) or 
    #    sufficiently large remainders (r >> lambda) that are likely to be used.
    
    remaining_after = bins - item
    feasible = remaining_after >= 0
    
    if not np.any(feasible):
        # No feasible bins; return all -inf
        return np.full(len(bins), -np.inf)
    
    scores = np.full(len(bins), -np.inf)
    
    # Get feasible gaps
    feasible_gaps = remaining_after[feasible]
    
    # 1. Base Score: Best Fit
    # We want to minimize the gap. So base score is -gap.
    base_score = -feasible_gaps
    
    # 2. Dead Space Penalty
    # Calculate D(r) = r * exp(-(r/lambda)^k)
    # Clip r to avoid numerical issues, though r >= 0 is guaranteed
    r = feasible_gaps
    # Avoid division by zero or negative issues, though lambda > 0
    ratio = r / _lambda_scale
    # Clip ratio to prevent overflow in exp
    ratio = np.clip(ratio, 0, 10)
    survival = np.exp(-np.power(ratio, _k))
    dead_space = r * survival
    
    # Weight for the dead space penalty
    # The magnitude of dead_space can vary. 
    # For r ~ 25 (lambda), D(25) = 25 * exp(-1) ~ 9.2.
    # For r ~ 50, D(50) = 50 * exp(-4) ~ 0.9.
    # For r ~ 5, D(5) = 5 * exp(-0.04) ~ 4.8.
    # The peak is around r = lambda * (1/(k-1))^(1/(k-1)) for k>1? 
    # Actually, max of x * exp(-(x/L)^k) is at x = L * (1/(k-1))^(1/(k-1)).
    # For k=2, max at x = L * 1 = 25. Value ~ 9.2.
    # Base score range: 0 to -100.
    # A penalty of ~10 is significant but not dominant.
    # We want to strongly avoid the dead zone.
    dead_weight = 1.0
    
    final_scores = base_score - dead_weight * dead_space
    
    scores[feasible] = final_scores
    
    return scores
# EVOLVE-BLOCK-END
