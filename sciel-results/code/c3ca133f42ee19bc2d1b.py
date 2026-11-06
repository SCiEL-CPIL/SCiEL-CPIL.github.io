import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 4096
    max_sum = 3 * n
    best_set = []
    
    # Use numpy boolean array for fast collision checks
    used_sums = np.zeros(max_sum + 1, dtype=bool)
    
    def get_new_sums_indices(x, S):
        """
        Generate indices of new sums involving x and elements of S.
        Returns None if collision, else list of new sum indices.
        """
        S_arr = np.array(S)
        
        # 3x
        s = 3 * x
        if used_sums[s]:
            return None
        new_sums = [s]
        
        # 2x + S
        if len(S) > 0:
            sums_2x = 2 * x + S_arr
            if np.any(used_sums[sums_2x]):
                return None
            new_sums.extend(sums_2x.tolist())
            
            # x + S_i + S_j
            # Compute all pair sums
            pair_sums = S_arr[:, None] + S_arr[None, :]
            new_pair_sums = x + pair_sums
            if np.any(used_sums[new_pair_sums]):
                return None
            new_sums.extend(new_pair_sums.tolist())
        else:
            # If S is empty, only 3x was checked
            pass
            
        return new_sums

    def mark_sums(sums):
        if sums:
            used_sums[sums] = True

    def unmark_sums(sums):
        if sums:
            used_sums[sums] = False

    def reset_sums():
        used_sums[:] = False

    def greedy_build(order):
        reset_sums()
        S = []
        for x in order:
            new_sums = get_new_sums_indices(x, S)
            if new_sums is not None:
                S.append(x)
                mark_sums(new_sums)
        return S

    # 1. Deterministic Greedy Ascending
    S1 = greedy_build(range(1, n + 1))
    if len(S1) > len(best_set):
        best_set = S1

    # 2. Deterministic Greedy Descending
    S2 = greedy_build(range(n, 0, -1))
    if len(S2) > len(best_set):
        best_set = S2

    # 3. Randomized Greedy
    # Use NumPy random generator for speed
    rng = np.random.default_rng(42)
    
    # We can run many more iterations now
    # Estimate: 50x speedup. 
    # Old: 2000 iters. New: 10000-20000 iters.
    max_iterations = 15000
    
    for i in range(max_iterations):
        if time.time() > deadline - 2.0:
            break
            
        # Random permutation
        # permutation is O(n log n) in C
        candidates = rng.permutation(n) + 1
        S = greedy_build(candidates)
        
        if len(S) > len(best_set):
            best_set = S
            # If we find a very good set, we might want to try to improve it locally?
            # But simple greedy is robust.
            pass

    return best_set
