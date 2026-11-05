import time
import random
from itertools import combinations

def _is_sidon(s):
    n = len(s)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            val = s[i] + s[j]
            if val in seen:
                return False
            seen.add(val)
    return True

def _max_sidon_subset_size(s):
    n = len(s)
    for k in range(n, 0, -1):
        for subset in combinations(range(n), k):
            sub = [s[i] for i in subset]
            if _is_sidon(sub):
                return k
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 70
    
    best_set = list(range(12))
    best_max_sidon = _max_sidon_subset_size(best_set)
    
    random.seed(42)
    
    # Structured candidates with many collisions
    structured = [
        list(range(12)),
        [0,1,2,3,4,5,6,7,8,9,10,11],
        [0,1,2,3,4,5,6,7,8,9,10,12],
        [0,1,2,3,4,5,6,7,8,10,11,12],
        [1,2,3,4,5,6,7,8,9,10,11,12],
        [0,1,2,3,4,5,6,7,8,9,11,12],
        [0,1,2,3,4,5,6,7,8,10,11,13],
        [0,1,2,3,4,5,6,7,9,10,11,12],
    ]
    
    for cand in structured:
        if time.time() > deadline - 10:
            break
        ms = _max_sidon_subset_size(cand)
        if ms < best_max_sidon:
            best_max_sidon = ms
            best_set = list(cand)
    
    # Random search with small ranges
    for _ in range(40000):
        if time.time() > deadline - 10:
            break
        base = random.randint(0, 110)
        spread = random.randint(12, 30)
        lo = max(0, base)
        hi = min(128, base + spread)
        if hi - lo + 1 < 12:
            continue
        cand = sorted(random.sample(range(lo, hi + 1), 12))
        ms = _max_sidon_subset_size(cand)
        if ms < best_max_sidon:
            best_max_sidon = ms
            best_set = list(cand)
    
    # Local search
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        s = sorted(best_set)
        for i in range(12):
            if time.time() > deadline - 3:
                break
            for new_val in range(129):
                if new_val in s:
                    continue
                trial = s[:i] + [new_val] + s[i+1:]
                trial.sort()
                ms = _max_sidon_subset_size(trial)
                if ms < best_max_sidon:
                    best_max_sidon = ms
                    best_set = list(trial)
                    improved = True
                    break
    
    return best_set
