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

# Polarization parameter: how much to penalize "middle" remainders
CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin. The item goes to the highest-scoring bin.
    
    Strategy: Combine the 'Dead Space' insight (avoiding medium-sized remainders 
    that are too small for typical future items but too large to be negligible) 
    with a consolidation bias and a polarization penalty.
    
    The core idea is that a remainder `r` has an expected "waste" or "dead space" 
    that is non-monotonic. Very small `r` is good (tight fit). Very large `r` is 
    bad (fragmentation), but medium `r` can be particularly bad if it falls in a 
    'gap' where most future items are larger than `r`.
    
    We model the penalty for a remainder `r` as:
    1. Dead Space: r * S(r), where S(r) is the survival function P(X > r).
       This term is high when `r` is medium-sized and many future items are larger.
       It is low when `r` is very small (tight fit) or very large (S(r) ~ 0).
    2. Consolidation Bias: A small linear penalty on `r` to encourage using 
       smaller bins when dead space is comparable, reducing the number of open bins.
    3. Polarization Penalty: A penalty for "middle" remainders (around CAPACITY/2)
       because half-full bins are less flexible than nearly-full or nearly-empty bins.
    
    We subtract this penalty from a base score to get the final score (higher is better).
    """
    # Check if we're close to the deadline
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        # Emergency: return a simple best-fit fallback
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores
    
    # Calculate remaining capacity
    remaining = bins - item
    
    # Identify valid bins (enough capacity)
    valid = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
        
    valid_indices = np.where(valid)[0]
    new_rems = remaining[valid]
    
    # Compute survival function: S(r) = exp(-(r/lambda)^k)
    # We use the estimated Weibull parameters.
    scaled = new_rems / _lambda_scale
    survival = np.exp(-np.power(scaled, _k))
    
    # Dead Space Penalty:
    # This term is high when `r` is in the "danger zone" where it's too small for 
    # most future items (high S(r)) but not small enough to be negligible (r > 0).
    dead_space = new_rems * survival
    
    # Consolidation Bias:
    # A small linear term to prefer smaller remainders, breaking ties in dead space
    # and generally encouraging bin closure.
    consolidation = 0.05 * new_rems
    
    # Polarization penalty (from Inspiration 2)
    # min(rem_after, CAPACITY - rem_after) measures "how close to the middle"
    # We want to MINIMIZE this, so we subtract it from the score
    pol_penalty = np.minimum(new_rems, CAPACITY - new_rems)
    
    # Weight for polarization penalty
    # Too high and it dominates; too low and it has no effect
    # The current program scores around -gap * multiplier
    # For gap ~ 20, multiplier ~ 2, base_score ~ -40
    # pol_penalty for gap=20 is min(20, 80) = 20
    # So a weight of 0.5 would give -10 penalty, which is significant but not dominant
    pol_weight = 0.3
    
    # Total penalty
    penalty = dead_space + consolidation + pol_weight * pol_penalty
    
    # Score is negative penalty (higher is better)
    scores_valid = -penalty
    
    # Small tie-breaker: prefer lower index for stability
    scores_valid -= np.arange(len(valid_indices)) * 1e-10
    
    scores[valid] = scores_valid
    
    return scores

# EVOLVE-BLOCK-END
