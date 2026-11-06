"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _is_sidon(subset):
    """Check if a subset is Sidon: all unordered pair sums (including diagonals) are unique."""
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(arr):
    """Find the size of the largest Sidon subset of arr."""
    n = len(arr)
    # Try from largest possible down
    for size in range(n, 0, -1):
        for combo in combinations(range(n), size):
            subset = [arr[i] for i in combo]
            if _is_sidon(subset):
                return size
    return 0

def construct():
    deadline = time.time() + 80  # Leave 10s margin for 90s limit
    best_set = list(range(12))
    best_score = _max_sidon_size(best_set)
    
    # Try some structured initial sets
    initial_candidates = [
        list(range(0, 12)),
        [0, 1, 2, 4, 8, 16, 32, 64, 96, 120, 124, 128],
        [0, 1, 3, 7, 15, 31, 63, 95, 127, 120, 110, 100],
        [i * 11 for i in range(12)],
        [i * 10 for i in range(12)],
        [i * 12 for i in range(12)],
    ]
    
    for cand in initial_candidates:
        if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
            score = _max_sidon_size(cand)
            if score < best_score:
                best_score = score
                best_set = cand[:]
    
    # Local search
    current = best_set[:]
    current_score = best_score
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Try random mutations
        improved = False
        for _ in range(100):
            if time.time() > deadline:
                break
            trial = current[:]
            # Pick a random index and change it
            idx = rng.randint(0, 11)
            new_val = rng.randint(0, 128)
            if new_val in trial:
                continue
            trial[idx] = new_val
            if len(set(trial)) != 12:
                continue
            
            score = _max_sidon_size(trial)
            if score < current_score:
                current = trial
                current_score = score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]
        
        if not improved:
            # Try a different random starting point
            trial = [rng.randint(0, 128) for _ in range(12)]
            # Ensure distinct
            while len(set(trial)) < 12:
                trial[rng.randint(0, 11)] = rng.randint(0, 128)
            score = _max_sidon_size(trial)
            if score < current_score:
                current = trial
                current_score = score
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]
    
    return best_set
# EVOLVE-BLOCK-END
