import math
import time

_start_time = None
_DEADLINE_FRACTION = 0.95

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Prioritize vectors with 3 distinct symbols (most "generic").
    2. Among those, prioritize vectors with symbol counts close to uniform (n/3).
    3. Use a hash for deterministic tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Check deadline
    # if time.time() - _start_time > 330 * _DEADLINE_FRACTION:
    #     # This is a priority function, it doesn't control the loop, 
    #     # but we can ensure it's fast.
    #     pass

    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
            
    # Distinct count
    counts = [c0, c1, c2]
    distinct_count = sum(1 for c in counts if c > 0)
    
    # Balance score (lower deviation is better)
    ideal = n / 3.0
    if ideal > 0:
        chi2 = ((c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2) / ideal
    else:
        chi2 = 0.0
        
    balance_score = -chi2
    
    # Primary: Distinct count (3 is best)
    # Secondary: Balance
    # Tertiary: Hash
    
    h = 0
    base = 31
    for i, val in enumerate(vector):
        h = (h * base + val + i * 7) % (10**9 + 7)
    
    hash_score = h / (10**9 + 7)
    
    # Scale distinct count to dominate
    # 3 distinct: +30000
    # 2 distinct: +20000
    # 1 distinct: +10000
    distinct_score = distinct_count * 10000.0
    
    # Balance score is in range [0, ~n]. Scale it to be significant but secondary.
    # Max chi2 is roughly n (if all in one bin).
    # So balance_score is in [-n, 0].
    # We want it to be within the distinct count gaps (10000).
    # If n is small (e.g., 10), chi2 is small.
    # If n is large (e.g., 100), chi2 can be up to 100.
    # 100 * 100 = 10000. This might overlap with distinct count gap.
    # Let's scale balance by 10.
    balance_scaled = balance_score * 10.0
    
    return float(distinct_score + balance_scaled + hash_score)
