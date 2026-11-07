"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _is_valid(s):
    """Check if set s satisfies cardinality-separated subset sum property."""
    n = len(s)
    if n == 0:
        return True
    # reachable[k] = set of sums achievable with exactly k elements
    reachable = [set() for _ in range(n + 1)]
    reachable[0].add(0)
    for x in s:
        for k in range(n, 0, -1):
            if reachable[k-1]:
                new_sums = {sm + x for sm in reachable[k-1]}
                reachable[k] = reachable[k] | new_sums
    # Check cross-cardinality: no sum appears in two different cardinality sets
    seen = {}
    for k in range(n + 1):
        for sm in reachable[k]:
            if sm in seen:
                return False
            seen[sm] = k
    return True

def _try_add(current_sums, current_set, x):
    """Try to add x to current_set. Returns (new_sums, True) if valid, else (None, False)."""
    n = len(current_set)
    # current_sums[k] is the set of sums with exactly k elements from current_set
    # After adding x, new cardinality goes up to n+1
    new_sums = [set() for _ in range(n + 2)]
    # Copy old sums
    for k in range(n + 1):
        new_sums[k] = set(current_sums[k])
    # Add new sums: for each k from 1 to n+1, new sums come from old (k-1) + x
    for k in range(1, n + 2):
        if k - 1 <= n and current_sums[k-1]:
            for sm in current_sums[k-1]:
                new_sums[k].add(sm + x)
    # Check cross-cardinality
    seen = {}
    for k in range(n + 2):
        for sm in new_sums[k]:
            if sm in seen:
                return None, False
            seen[sm] = k
    return new_sums, True

def _greedy_build(pool, order=None):
    """Greedy build: try to add elements from pool in given order."""
    if order is None:
        order = list(pool)
    
    current_set = []
    # Initialize: sums[0] = {0}
    current_sums = [set() for _ in range(1)]
    current_sums[0].add(0)
    
    for x in order:
        if x in current_set:
            continue
        new_sums, ok = _try_add(current_sums, current_set, x)
        if ok:
            current_set.append(x)
            current_sums = new_sums
    
    return current_set

def construct():
    deadline = time.time() + 85  # 85s deadline, 5s safety margin
    
    best = list(range(128, 136))  # 8 elements, known valid fallback
    best_len = len(best)
    
    # Try deterministic strategies first
    # Strategy 1: smallest first
    pool = list(range(1, 257))
    result = _greedy_build(pool, sorted(pool))
    if len(result) > best_len:
        best = result
        best_len = len(result)
    
    # Strategy 2: largest first
    result = _greedy_build(pool, sorted(pool, reverse=True))
    if len(result) > best_len:
        best = result
        best_len = len(result)
    
    # Strategy 3: try superincreasing-like: 1, 2, 4, 8, ... but capped at 256
    superinc = []
    s = 0
    for i in range(1, 257):
        if i > s:
            superinc.append(i)
            s += i
        if len(superinc) >= 60:
            break
    if superinc:
        result = _greedy_build(pool, superinc)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Randomized greedy with multiple restarts
    rng = random.Random(42)
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Random permutation of pool
        shuffled = pool[:]
        rng.shuffle(shuffled)
        
        # Also try biased: mix of small and large
        if restart % 3 == 0:
            # Prefer smaller values
            shuffled = sorted(shuffled, key=lambda x: x + rng.random() * 100)
        elif restart % 3 == 1:
            # Prefer larger values  
            shuffled = sorted(shuffled, key=lambda x: -x + rng.random() * 100)
        # else: fully random
        
        result = _greedy_build(pool, shuffled)
        if len(result) > best_len:
            best = result
            best_len = len(result)
            # Try to extend with remaining elements
            remaining = [x for x in pool if x not in best]
            rng.shuffle(remaining)
            # Rebuild with best first then remaining
            order = list(best) + remaining
            result2 = _greedy_build(pool, order)
            if len(result2) > best_len:
                best = result2
                best_len = len(result2)
    
    # Final: try to extend best set
    if best_len < 60:
        remaining = [x for x in range(1, 257) if x not in best]
        rng.shuffle(remaining)
        # Build fresh with best elements first
        order = list(best) + remaining
        result = _greedy_build(pool, order)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Ensure return type is correct
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
