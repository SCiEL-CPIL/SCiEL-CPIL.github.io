import time
import random
import itertools

def is_sidon(subset):
    """Check if subset has all unordered pair sums (i<=j) unique."""
    s = sorted(subset)
    n = len(s)
    seen_sums = set()
    for i in range(n):
        for j in range(i, n):
            sm = s[i] + s[j]
            if sm in seen_sums:
                return False
            seen_sums.add(sm)
    return True

def max_sidon_size(S):
    """Find largest subset of S that is Sidon (all pair sums i<=j unique)."""
    S = sorted(S)
    for k in range(len(S), 0, -1):
        for comb in itertools.combinations(S, k):
            if is_sidon(list(comb)):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Contiguous set 0-11: pair sums range from 0 to 22 (23 values)
    # For a Sidon set of size k, need k(k+1)/2 distinct sums
    # k=6: 21 sums <= 23, possible. k=7: 28 > 23, impossible.
    # So max Sidon subset of [0,11] is at most 6.
    
    # Try various candidate ambient sets
    candidates = [
        list(range(12)),                    # 0-11
        list(range(0, 24, 2)),              # 0,2,4,...,22 (same structure, scaled)
        list(range(0, 36, 3)),              # 0,3,6,...,33
        [i*10 for i in range(12)],          # 0,10,20,...,110
        [0, 1, 2, 4, 8, 16, 32, 64, 96, 112, 120, 124],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13],
        list(range(12, 24)),                # 12-23
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
    ]
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(sorted(c))
    
    current_best = unique_candidates[0]
    current_best_val = max_sidon_size(current_best)
    
    for c in unique_candidates[1:]:
        if time.time() > deadline:
            break
        val = max_sidon_size(c)
        if val < current_best_val:
            current_best_val = val
            current_best = c
    
    # Random search around current best
    if time.time() < deadline - 5:
        for _ in range(200):
            if time.time() > deadline:
                break
            cand = list(current_best)
            # Randomly perturb 1-3 elements
            n_perturb = random.randint(1, 3)
            for _ in range(n_perturb):
                idx = random.randint(0, 11)
                new_val = random.randint(0, 128)
                attempts = 0
                while new_val in cand and attempts < 30:
                    new_val = random.randint(0, 128)
                    attempts += 1
                if new_val in cand:
                    continue
                cand[idx] = new_val
            cand = sorted(set(cand))
            if len(cand) != 12:
                continue
            
            val = max_sidon_size(cand)
            if val < current_best_val:
                current_best_val = val
                current_best = cand
    
    return current_best
