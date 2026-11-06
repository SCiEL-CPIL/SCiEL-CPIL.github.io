import time
import random
import itertools

def is_sidon(subset):
    """Check if subset has all pair sums (i<=j) unique."""
    s = sorted(subset)
    n = len(s)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            sm = s[i] + s[j]
            if sm in sums:
                return False
            sums.add(sm)
    return True

def max_sidon_size(S):
    """Find largest Sidon subset of S."""
    for k in range(len(S), 0, -1):
        for comb in itertools.combinations(S, k):
            if is_sidon(list(comb)):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    # Test structured candidates that have abundant sum collisions
    candidates = []
    
    # Contiguous
    candidates.append(list(range(12)))
    
    # APs with various differences and starts
    for d in range(2, 13):
        max_start = 128 - 11 * d
        for start in range(0, max_start + 1):
            cand = [start + i * d for i in range(12)]
            candidates.append(cand)
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    current_best = None
    current_best_val = 13
    
    for c in unique_candidates:
        if time.time() > deadline:
            break
        if len(c) != 12 or len(set(c)) != 12:
            continue
        val = max_sidon_size(c)
        if val < current_best_val:
            current_best_val = val
            current_best = sorted(c)
    
    if current_best is None:
        current_best = list(range(12))
        current_best_val = max_sidon_size(current_best)
    
    return current_best
