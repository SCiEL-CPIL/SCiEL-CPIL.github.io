import numpy as np
import time
from math import isqrt

# Global state
_start_time = None

# Deadline guard threshold (seconds)
_DEADLINE = 145.0

# Precompute smallest prime factor (SPF) for fast factorization
# We only need to factor remainders which are < 100.
# Max possible remainder is 100 (if item is 0, though items are positive).
# Actually, bins start at 100. Item > 0. So remainder < 100.
# Let's precompute up to 100 to be safe.
_MAX_R = 100
_spf = list(range(_MAX_R + 1))
for i in range(2, isqrt(_MAX_R) + 1):
    if _spf[i] == i:  # i is prime
        for j in range(i * i, _MAX_R + 1, i):
            if _spf[j] == j:
                _spf[j] = i

# Cache for the score of each possible remainder (integer)
# We map remainder float to int. Since items are Weibull, they are floats.
# The remainder will be a float.
# The hypothesis is about "integer-sized" items or arithmetic compatibility.
# We will round the remainder to the nearest integer for the factorization count.
# If remainder is 0, count is 0 (perfect fit, ideal).
# If remainder is 1, count is 0 (1 has no prime factors).
# If remainder is 2, count is 1.
# If remainder is 4, count is 2.
# If remainder is 6, count is 2.
# We prefer fewer factors.
# Also, we need to handle the "magnitude" of the remainder.
# A remainder of 10 with 2 factors (2*5) vs remainder of 4 with 2 factors (2*2).
# Remainder 4 is smaller, so it's "tighter".
# Pure factor count might ignore magnitude.
# Let's combine: score = - (factor_count + alpha * magnitude_normalized)
# Or simply: we want to minimize factor_count. If tied, minimize remainder.
# Let's precompute the factor count for integers 0..100.
_factor_count = [0] * (_MAX_R + 1)
for i in range(2, _MAX_R + 1):
    x = i
    count = 0
    while x > 1:
        count += 1
        x //= _spf[x]
    _factor_count[i] = count

# _factor_count[0] = 0
# _factor_count[1] = 0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _start_time
    
    # Initialize timing if not started
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback to simple Best Fit (minimize remainder)
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
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
    
    # Convert remainders to integers for factorization lookup
    # Round to nearest integer. 
    # If remainder is 0.4, int is 0. 
    # If remainder is 0.6, int is 1.
    # This discretization is the core of the "number-theoretic" approach.
    rem_ints = np.round(new_rems).astype(int)
    
    # Clip to valid range [0, _MAX_R]
    rem_ints = np.clip(rem_ints, 0, _MAX_R)
    
    # Get factor counts
    # _factor_count is a list, we can convert to array for vectorized lookup
    # Or just use it as a lookup table.
    # Since len(valid_indices) can be up to 1000, and we do this per item,
    # vectorized lookup is better.
    fc_array = np.array(_factor_count)
    counts = fc_array[rem_ints]
    
    # Score calculation:
    # We want to minimize the "complexity" of the remainder.
    # Primary metric: Factor count (lower is better).
    # Secondary metric: Remainder size (lower is better, for consolidation).
    # Tertiary metric: Stability (lower index).
    
    # Normalize remainder to 0..1 range for weighting?
    # Max remainder is ~100.
    # Let's use a weighted sum: penalty = counts * weight_c + (new_rems / 100.0) * weight_r
    # If we prioritize factor count heavily, we might pick a large remainder with simple factors
    # over a small remainder with complex factors.
    # E.g., Rem=100 (2^2*5^2, count=4) vs Rem=7 (prime, count=1).
    # Rem=7 is much better for packing (less waste).
    # So magnitude must be significant.
    
    # Let's try: penalty = counts + 0.1 * new_rems
    # This favors low factor counts but still penalizes large gaps.
    
    penalty = counts + 0.1 * new_rems
    
    # Score is negative penalty
    scores_valid = -penalty
    
    # Small tie-breaker
    scores_valid -= np.arange(len(valid_indices)) * 1e-10
    
    scores[valid] = scores_valid
    
    return scores
