"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(ambient_set):
    n = len(ambient_set)
    for k in range(n, 0, -1):
        for combo in combinations(ambient_set, k):
            if _is_sidon(combo):
                return k
    return 1

def construct():
    deadline = time.time() + 30
    rng = random.Random(42)
    
    best_set = list(range(12))
    best_score = _max_sidon_size(best_set)
    
    current = list(range(12))
    current_score = best_score
    
    while time.time() < deadline:
        candidate = list(current)
        idx = rng.randrange(12)
        new_val = rng.randint(0, 128)
        if new_val in candidate:
            continue
        candidate[idx] = new_val
        
        score = _max_sidon_size(candidate)
        
        if score < best_score:
            best_score = score
            best_set = list(candidate)
        
        if score < current_score:
            current = candidate
            current_score = score
        elif score == current_score and rng.random() < 0.2:
            current = candidate
    
    return best_set
# EVOLVE-BLOCK-END
