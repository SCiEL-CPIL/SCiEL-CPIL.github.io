"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.special

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 15.0  # seconds before the 150s limit
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

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

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Probabilistic Best-Fit with Fragmentation Penalty.
    
    We combine two insights:
    1. Best-Fit minimizes immediate waste (gap = bins[i] - item).
    2. However, creating very small gaps (fragments) is bad because they cannot
       accommodate future items (which have an expected size E[X]).
    
    We use a penalty mechanism that discourages bins where the remaining space
    after placing the current item is too small to likely fit the next item.
    
    Score = -gap * (1 + exp(-(gap - E[X]) / sigma))
    
    - If gap >> E[X], exp term is ~0, Score ~ -gap (Best Fit behavior).
    - If gap << E[X], exp term is large, Score becomes very negative (strong penalty).
    - This smoothly transitions between Best Fit and "Avoid Small Gaps".
    """
    # Check if we're close to the deadline
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        # Emergency: return a simple best-fit fallback
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores
    
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
    # Clip to avoid overflow/underflow issues, though exp handles it mostly
    arg = np.clip(arg, -10, 10)
    
    multiplier = 1.0 + np.exp(-arg)
    
    # Final score: negative gap scaled by multiplier
    # We want higher scores for better bins.
    # Best fit = small gap.
    # Penalty = large multiplier when gap is small.
    # So score = -gap * multiplier.
    # If gap is small (good fit), but < E[X], multiplier is huge -> very negative score.
    # If gap is large (bad fit), multiplier is ~1 -> score is just -gap.
    # Wait, if gap is large, -gap is very negative.
    # If gap is small and > E[X], multiplier is ~1, score is -gap (small negative).
    # This correctly prioritizes small gaps that are still "safe".
    
    scores[feasible] = -feasible_gaps * multiplier
    
    return scores
# EVOLVE-BLOCK-END
