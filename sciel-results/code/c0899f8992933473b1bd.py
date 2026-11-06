"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin for 90s limit
    
    best_size = 0
    best_set = []
    
    # We will try to build sets of size up to 80.
    # The seed found a small set. We expect to find much larger ones.
    # Let's try to maximize size.
    
    # To speed up, we can use a random ordering of candidates in [1, 4096].
    # Or we can use a smarter heuristic.
    # Random restarts with greedy selection based on random permutation of [1, 4096].
    
    # Optimization: 
    # When adding x to a set S, the new triple sums are:
    # 1. 3*x
    # 2. 2*x + s for s in S
    # 3. x + s1 + s2 for s1, s2 in S (s1 <= s2)
    # We need to check if any of these are in the existing set of sums.
    # And we must add all new sums to the set.
    
    # Let's use a set for sums.
    
    def can_add(S, x, sums_set):
        # S is a sorted list of selected numbers
        k = len(S)
        if k == 0:
            return True
        if k == 1:
            s0 = S[0]
            # Sums: 3x, 2x+s0, x+s0+s0
            if 3*x in sums_set: return False
            if 2*x + s0 in sums_set: return False
            if x + 2*s0 in sums_set: return False
            return True
        if k == 2:
            s0, s1 = S
            # Sums: 3x, 2x+s0, 2x+s1, x+s0+s0, x+s0+s1, x+s1+s1
            if 3*x in sums_set: return False
            if 2*x + s0 in sums_set: return False
            if 2*x + s1 in sums_set: return False
            if x + 2*s0 in sums_set: return False
            if x + s0 + s1 in sums_set: return False
            if x + 2*s1 in sums_set: return False
            return True
        
        # General case
        # 3x
        if 3*x in sums_set:
            return False
        
        # 2x + s
        for s in S:
            if 2*x + s in sums_set:
                return False
        
        # x + s1 + s2
        # To be efficient, we can iterate s1, s2
        # But O(k^2) might be too slow if k is large and we have many restarts?
        # k=80, k^2=6400. 4096 candidates * 6400 ops = 26M ops per restart.
        # In Python, this is too slow.
        # We need NumPy or a better check.
        
        # Let's use NumPy for the collision check.
        # S is a numpy array.
        # sums_to_check = np.array([3*x])
        # sums_to_check = np.concatenate([sums_to_check, 2*x + S])
        # # x + S[:, None] + S[None, :] -> flatten
        # # We only need unique pairs? No, the set of sums handles duplicates.
        # # But we must ensure we don't check (s1, s2) and (s2, s1) separately if we want to be fast?
        # # Actually, sum is commutative.
        # # Let's generate all pairs.
        
        # This Python loop might be acceptable if we limit restarts?
        # 26M ops in Python is ~1-2 seconds.
        # 80s / 2s = 40 restarts.
        # This is feasible.
        
        for i in range(k):
            s1 = S[i]
            # x + s1 + s2 for s2 >= s1
            for j in range(i, k):
                s2 = S[j]
                if x + s1 + s2 in sums_set:
                    return False
        return True

    def add_to_sums(S, x, sums_set):
        # Add new sums to the set
        k = len(S)
        sums_set.add(3*x)
        for s in S:
            sums_set.add(2*x + s)
        for i in range(k):
            s1 = S[i]
            for j in range(i, k):
                s2 = S[j]
                sums_set.add(x + s1 + s2)

    # To make it faster, we can pre-filter candidates?
    # Or just run many random restarts.
    
    # Let's try to improve the inner loop with NumPy for the general case.
    # If k > 2, use NumPy.
    
    def can_add_np(S_np, x, sums_set):
        # S_np is a 1D numpy array of int
        # Generate all new sums
        # 3*x
        # 2*x + S
        # x + S_i + S_j
        
        # This creates a large array.
        # For k=80, 2*x+S is size 80.
        # x + S_i + S_j is size 80*80/2 ~ 3200.
        # Total ~ 3300 integers.
        # Checking if any are in sums_set.
        # sums_set is a Python set.
        # We can convert sums_set to a numpy array? No, it changes every time.
        # We can just iterate?
        # Or use a boolean array of size 3*4096 = 12288?
        # Max sum is 3*4096 = 12288.
        # A boolean array of size 12289 is very fast to index.
        
        # Let's use a boolean array for sums_set.
        # This is a structural change in representation.
        
        pass

    # Revised strategy with Boolean Array for sums
    MAX_SUM = 3 * 4096
    sum_flag = np.zeros(MAX_SUM + 1, dtype=bool)
    
    def can_add_fast(S, x, sum_flag):
        # S is a list
        k = len(S)
        if 3*x > MAX_SUM: return False
        if sum_flag[3*x]: return False
        
        for s in S:
            val = 2*x + s
            if val > MAX_SUM or sum_flag[val]:
                return False
        
        for i in range(k):
            s1 = S[i]
            for j in range(i, k):
                s2 = S[j]
                val = x + s1 + s2
                if val > MAX_SUM or sum_flag[val]:
                    return False
        return True

    def add_fast(S, x, sum_flag):
        sum_flag[3*x] = True
        for s in S:
            sum_flag[2*x + s] = True
        for i in range(k):
            s1 = S[i]
            for j in range(i, k):
                s2 = S[j]
                sum_flag[x + s1 + s2] = True

    # We need to be careful with k in add_fast.
    def add_fast_correct(S, x, sum_flag):
        sum_flag[3*x] = True
        for s in S:
            sum_flag[2*x + s] = True
        k = len(S)
        for i in range(k):
            s1 = S[i]
            for j in range(i, k):
                s2 = S[j]
                sum_flag[x + s1 + s2] = True

    # Run multiple restarts
    # We can also try to bias the random selection towards larger numbers?
    # Or just random permutation.
    
    # To maximize cardinality, we should probably try to fill the space.
    # Greedy with random permutation is a standard approach.
    
    for restart in range(1000): # Limit restarts, but check time
        if time.time() > deadline:
            break
        
        random.seed(restart) # Reproducibility for debugging, but random is fine
        candidates = list(range(1, 4097))
        random.shuffle(candidates)
        
        S = []
        sum_flag = np.zeros(MAX_SUM + 1, dtype=bool)
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            if can_add_fast(S, x, sum_flag):
                S.append(x)
                add_fast_correct(S, x, sum_flag)
                
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S.copy()
                    # If we reach 80, we can't do better?
                    # The task says "Maximize cardinality". 80 is the cap.
                    # If we hit 80, we can return immediately?
                    # "At most 80 positive integers".
                    # So 80 is the max possible.
                    if best_size == 80:
                        return best_set
        
        # If we finished a restart, we might try to improve with local search?
        # But time is tight.
        
    return best_set
# EVOLVE-BLOCK-END
