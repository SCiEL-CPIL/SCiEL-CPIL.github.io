import time
import random
from itertools import combinations

N = 128
BIG = list(range(N // 2 + 1, N + 1))  # 65..128
SMALL = list(range(1, N // 2 + 1))    # 1..64

def _can_add(small_set, candidate):
    """Check if adding candidate creates a forbidden triple.
    Only need to check triples that include candidate."""
    s = small_set | {candidate}
    for a in small_set:
        for b in small_set:
            if a >= b:
                continue
            # Triple is (candidate, a, b)
            x, y, z = candidate, a, b
            xy = x + y
            xz = x + z
            yz = y + z
            if (xy in s or xy >= 65) and (xz in s or xz >= 65) and (yz in s or yz >= 65):
                return False
    return True

def _greedy_build(order):
    small_set = set()
    for elem in order:
        if _can_add(small_set, elem):
            small_set.add(elem)
    return small_set

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_set = set()
    
    # Initial greedy with natural order
    best_set = _greedy_build(list(SMALL))
    
    random.seed(42)
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        order = list(SMALL)
        random.shuffle(order)
        
        candidate = _greedy_build(order)
        
        if len(candidate) > len(best_set):
            best_set = candidate
    
    result = list(BIG) + list(best_set)
    result.sort()
    
    # Final validation
    s = set(result)
    for a, b, c in combinations(result, 3):
        if (a + b) in s and (a + c) in s and (b + c) in s:
            return list(BIG)
    
    return result
