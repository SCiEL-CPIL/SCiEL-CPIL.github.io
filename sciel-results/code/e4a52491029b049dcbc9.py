"""Cap set greedy builder with distinct-count-dominant priority."""
import time
from collections import Counter

# EVOLVE-BLOCK-START
_start_time = None
_DEADLINE = 313.0  # ~95% of 330s, safety margin

def priority(vector, n):
    """Priority for greedy cap set construction.
    
    Strategy (combined from current + best inspiration):
    1. PRIMARY: Distinct symbol count (3 >> 2 >> 1) with 10000x scale gap.
       Vectors using all three symbols are far more "generic" and less likely
       to form affine lines with each other.
    2. SECONDARY: Smooth balance — minimize sum of squared deviations from n/3.
       Among vectors with the same distinct count, more uniform distributions
       are preferred (smoother surrogate than L1 for ranking).
    3. TERTIARY: Coordinate sum mod 3 penalty — vectors with sum ≡ 0 mod 3
       are slightly more likely to form lines, so penalize them.
    4. QUATERNARY: Fast structural hash for deterministic tie-breaking.
    
    Optimized for speed: Counter (C-level) + minimal arithmetic.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Fast symbol counting using Counter (C-implemented)
    c = Counter(vector)
    c0 = c.get(0, 0)
    c1 = c.get(1, 0)
    c2 = c.get(2, 0)
    
    # PRIMARY: Distinct count — must dominate all other signals
    distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    # Scale: 3 distinct = 30000, 2 distinct = 20000, 1 distinct = 10000
    # This ensures no balance/hash signal can cross a distinct-count boundary
    distinct_score = distinct * 10000.0
    
    # SECONDARY: Smooth balance via sum of squared deviations from n/3
    ideal = n / 3.0
    sq_dev = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    # Scale so max deviation (~n for all-in-one-bin) stays well below 10000 gap
    balance_score = -sq_dev * 5.0
    
    # TERTIARY: Coordinate sum mod 3 penalty
    coord_sum = (c0 * 0 + c1 * 1 + c2 * 2) % 3
    sum_penalty = 2.0 if coord_sum == 0 else 0.0
    
    # QUATERNARY: Fast polynomial rolling hash for tie-breaking
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF  # in [0, 1)
    
    return float(distinct_score + balance_score - sum_penalty + hash_score)

# EVOLVE-BLOCK-END
