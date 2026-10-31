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
        
    # Strategy: Probabilistic Best-Fit with Polarization
    #
    # We combine two structural insights:
    # 1. Probabilistic penalty: Discourage creating very small gaps that cannot
    #    accommodate future items (which have an expected size E[X]). This prevents
    #    "dead space" that forces opening new bins.
    # 2. Polarization penalty: Discourage "middle" remainders (around CAPACITY/2)
    #    because half-full bins are less flexible than nearly-full or nearly-empty bins.
    #    This encourages bins to become either full (0) or empty (100), reducing
    #    the number of "half-full" bins that are hard to pack.
    #
    # Score = -gap * multiplier - polarization_penalty
    #
    # Where:
    # - multiplier = 1 + exp(-(gap - E[X]) / sigma) penalizes small gaps
    # - polarization_penalty = min(rem_after, CAPACITY - rem_after) penalizes middle states
    
    remaining_after = bins - item
    feasible = remaining_after >= 0
    
    if not np.any(feasible):
        # No feasible bins; return all -inf
        return np.full(len(bins), -np.inf)
    
    scores = np.full(len(bins), -np.inf)
    
    # Get feasible gaps
    feasible_gaps = remaining_after[feasible]
    
    # Calculate the probabilistic multiplier
    # sigma controls the sharpness of the transition
    sigma = 5.0
    
    # Argument for exponential
    arg = (feasible_gaps - _EXPECTED_NEXT_ITEM) / sigma
    # Clip to avoid overflow/underflow issues
    arg = np.clip(arg, -10, 10)
    
    multiplier = 1.0 + np.exp(-arg)
    
    # Base score: negative gap scaled by multiplier
    # This is a "weighted best fit" where the weight is higher for small gaps
    # that are close to the expected next item size.
    base_score = -feasible_gaps * multiplier
    
    # Polarization penalty
    # min(rem_after, CAPACITY - rem_after) measures "how close to the middle"
    # We want to MINIMIZE this, so we subtract it from the score
    pol_penalty = np.minimum(feasible_gaps, CAPACITY - feasible_gaps)
    
    # Weight for polarization penalty
    # Too high and it dominates; too low and it has no effect
    # For gap ~ 20, multiplier ~ 2, base_score ~ -40
    # pol_penalty for gap=20 is min(20, 80) = 20
    # A weight of 0.5 gives -10 penalty, which is significant but not dominant
    pol_weight = 0.5
    
    # Final score
    final_scores = base_score - pol_weight * pol_penalty
    
    scores[feasible] = final_scores
    
    return scores
# EVOLVE-BLOCK-END
