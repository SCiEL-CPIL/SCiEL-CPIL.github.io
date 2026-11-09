"""Optimized construction for 32 integers in 1..512 maximizing distinct 3-AP differences."""
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    def score_set(S):
        """Count distinct d>0 such that a, a+d, a+2d all in S."""
        S_set = set(S)
        diffs = set()
        n = len(S)
        for i in range(n):
            x = S[i]
            for j in range(i + 1, n):
                z = S[j]
                if (x + z) % 2 == 0:
                    mid = (x + z) // 2
                    if mid in S_set:
                        d = (z - x) // 2
                        if d > 0:
                            diffs.add(d)
        return len(diffs)

    # Initial candidates
    candidates = []
    candidates.append(list(range(1, 33)))
    
    # Random starts
    for _ in range(50):
        random.seed(int(time.time() * 1000) + _)
        cand = random.sample(range(1, 513), 32)
        candidates.append(cand)

    best_set = None
    best_score = 0
    for cand in candidates:
        if len(cand) == 32 and all(1 <= x <= 512 for x in cand) and len(set(cand)) == 32:
            s = score_set(cand)
            if s > best_score:
                best_score = s
                best_set = cand[:]

    if best_set is None:
        best_set = list(range(1, 33))
        best_score = score_set(best_set)

    random.seed(42)
    
    current_set = best_set[:]
    current_score = best_score
    best_set_sa = best_set[:]
    best_score_sa = best_score

    max_iters = 5000
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # 1. Local Search Phase
        # Try to improve by single swaps
        improved = True
        while improved:
            improved = False
            S_set = set(current_set)
            # Shuffle indices to randomize order
            indices = list(range(32))
            random.shuffle(indices)
            
            for idx in indices:
                rem_val = current_set[idx]
                # Try a few random additions
                # To be fast, we don't try all 480 additions for every removal.
                # We try 20 random additions.
                for _ in range(20):
                    add_val = random.randint(1, 512)
                    if add_val in S_set or add_val == rem_val:
                        continue
                    
                    # Perform swap
                    new_set = current_set[:]
                    new_set[idx] = add_val
                    new_score = score_set(new_set)
                    
                    if new_score > current_score:
                        current_set = new_set
                        current_score = new_score
                        S_set = set(current_set)
                        improved = True
                        if current_score > best_score_sa:
                            best_score_sa = current_score
                            best_set_sa = current_set[:]
                        break # Restart local search with new state
        
        # 2. Perturbation Phase
        # Randomly swap 2-4 elements
        perturbed_set = current_set[:]
        for _ in range(random.randint(2, 4)):
            i = random.randint(0, 31)
            j = random.randint(0, 31)
            # Swap i and j? No, remove one and add random.
            # Better: remove random element, add random element
            rem_idx = random.randint(0, 31)
            rem_val = perturbed_set[rem_idx]
            add_val = random.randint(1, 512)
            if add_val != rem_val and add_val not in set(perturbed_set):
                perturbed_set[rem_idx] = add_val
        
        current_set = perturbed_set
        current_score = score_set(current_set)
        
        if current_score > best_score_sa:
            best_score_sa = current_score
            best_set_sa = current_set[:]

    return best_set_sa
