import time
from collections import Counter

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Primary: Prioritize vectors with all 3 distinct symbols.
    2. Secondary: Among those, prioritize vectors with symbol counts close to uniform (L1 deviation).
    3. Tertiary: Mix score (count of position changes) + fast hash for tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    c = Counter(vector)
    c0 = c.get(0, 0)
    c1 = c.get(1, 0)
    c2 = c.get(2, 0)
    
    if c0 > 0 and c1 > 0 and c2 > 0:
        distinct_score = 30000.0
    elif c0 > 0 and c1 > 0:
        distinct_score = 20000.0
    elif c0 > 0 and c2 > 0:
        distinct_score = 20000.0
    elif c1 > 0 and c2 > 0:
        distinct_score = 20000.0
    else:
        distinct_score = 10000.0
    
    ideal = n / 3.0
    dev = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    balance_score = -dev * 10.0
    
    # Mix score: count position changes (more mixed = better)
    changes = 0
    if n > 1:
        prev = vector[0]
        for val in vector[1:]:
            if val != prev:
                changes += 1
            prev = val
    mix_score = changes * 0.5
    
    # Fast hash
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    
    hash_score = h / 0x7FFFFFFF
    
    return float(distinct_score + balance_score + mix_score + hash_score)
