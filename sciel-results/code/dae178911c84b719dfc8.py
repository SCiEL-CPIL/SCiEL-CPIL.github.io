"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _is_sidon_with_diagonals(subset):
    k = len(subset)
    seen = set()
    for i in range(k):
        for j in range(i, k):
            s = subset[i] + subset[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _largest_sidon_subset(arr):
    n = len(arr)
    arr = tuple(sorted(arr))
    for size in range(n, 0, -1):
        for combo in combinations(arr, size):
            if _is_sidon_with_diagonals(combo):
                return size
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_set = list(range(12))
    best_val = _largest_sidon_subset(best_set)
    
    random.seed(7)
    
    while time.time() < deadline:
        candidate = random.sample(range(129), 12)
        candidate.sort()
        
        val = _largest_sidon_subset(candidate)
        if val < best_val:
            best_set = candidate[:]
            best_val = val
            if best_val <= 4:
                break
        
        # Local search: mutate one element
        for _ in range(200):
            if time.time() > deadline:
                break
            idx = random.randint(0, 11)
            old = candidate[idx]
            new_val = random.randint(0, 128)
            if new_val == old:
                continue
            if new_val in candidate:
                continue
            candidate[idx] = new_val
            candidate.sort()
            val = _largest_sidon_subset(candidate)
            if val < best_val:
                best_set = candidate[:]
                best_val = val
                if best_val <= 4:
                    return best_set
            else:
                candidate[idx] = old
                candidate.sort()
    
    return best_set
# EVOLVE-BLOCK-END

