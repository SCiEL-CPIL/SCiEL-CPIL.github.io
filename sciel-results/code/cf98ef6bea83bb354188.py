import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 4096
    best_set = []
    best_len = 0
    
    # Max possible triple sum: 3*4096 = 12288
    MAX_SUM = 3 * N
    
    # Helper to check if x can be added to current set S with current sums
    # Returns True if valid, and if so, the set of new sums to add
    def check_and_get_new_sums(S, sums_bitset, x):
        n = len(S)
        
        # 1. 3x
        s_3x = 3 * x
        if s_3x <= MAX_SUM and sums_bitset[s_3x]:
            return False, None
        
        # 2. 2x + s_i
        for s_i in S:
            s_val = 2 * x + s_i
            if s_val <= MAX_SUM and sums_bitset[s_val]:
                return False, None
        
        # 3. x + s_i + s_j (i <= j)
        for i in range(n):
            s_i = S[i]
            for j in range(i, n):
                s_val = x + s_i + S[j]
                if s_val <= MAX_SUM and sums_bitset[s_val]:
                    return False, None
        
        # All checks passed; return the new sums as a set
        new_sums_set = set()
        new_sums_set.add(s_3x)
        for s_i in S:
            new_sums_set.add(2 * x + s_i)
        for i in range(n):
            for j in range(i, n):
                new_sums_set.add(x + S[i] + S[j])
        
        return True, new_sums_set

    def run_greedy(order):
        S = []
        sums_bitset = np.zeros(MAX_SUM + 1, dtype=bool)
        
        for x in order:
            valid, new_sums = check_and_get_new_sums(S, sums_bitset, x)
            if valid:
                S.append(x)
                for s in new_sums:
                    sums_bitset[s] = True
                
        return S

    # Strategy 1: Pure Greedy with Natural Order
    # This is fast and gives a baseline.
    S_init = run_greedy(range(1, N + 1))
    if len(S_init) > best_len:
        best_set = S_init
        best_len = len(S_init)
        
    # Strategy 2: Randomized Greedy with Multiple Restarts
    # We try to find a larger set by shuffling the order.
    # Since the constraint is global, the order matters.
    # Larger numbers tend to have fewer collisions with small numbers? 
    # Actually, small numbers collide with each other more often in sums?
    # Let's try different random seeds and biases.
    
    random.seed(42)
    
    # Time budget for randomized search
    # Each greedy pass is O(N * K^2) where K is current size. 
    # N=4096, K~30-50. 4096 * 2500 ~ 10^7 operations. 
    # Python might take 1-2 seconds per pass.
    # We have ~50 seconds. So we can do ~20-30 passes.
    
    while time.time() < deadline - 5.0:
        # Create a random order
        # Sometimes biasing towards larger numbers helps avoid early collisions?
        # Or just uniform random.
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S_try = run_greedy(order)
        
        if len(S_try) > best_len:
            best_set = S_try
            best_len = len(S_try)
            
    # Strategy 3: Local Improvement on Best Set
    # Try to add more elements to the best set found so far.
    if best_len > 0:
        S_cur = list(best_set)
        sums_bitset = np.zeros(MAX_SUM + 1, dtype=bool)
        # Recompute sums for current best set
        n = len(S_cur)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    s_val = S_cur[i] + S_cur[j] + S_cur[k]
                    if s_val <= MAX_SUM:
                        sums_bitset[s_val] = True
                    
        S_cur_set = set(S_cur)
        
        # Try to add any remaining number
        for x in range(1, N + 1):
            if time.time() > deadline - 2.0:
                break
            if x in S_cur_set:
                continue
            valid, new_sums = check_and_get_new_sums(S_cur, sums_bitset, x)
            if valid:
                S_cur.append(x)
                S_cur_set.add(x)
                for s in new_sums:
                    sums_bitset[s] = True
                if len(S_cur) > best_len:
                    best_set = list(S_cur)
                    best_len = len(S_cur)
                    
    # Strategy 4: Swap-based local improvement
    if best_len > 0 and time.time() < deadline - 3.0:
        S_cur = list(best_set)
        sums_bitset = np.zeros(MAX_SUM + 1, dtype=bool)
        n = len(S_cur)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    s_val = S_cur[i] + S_cur[j] + S_cur[k]
                    if s_val <= MAX_SUM:
                        sums_bitset[s_val] = True
        
        S_cur_set = set(S_cur)
        candidates = [x for x in range(1, N + 1) if x not in S_cur_set]
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for idx in range(len(S_cur)):
                if time.time() > deadline - 2.0:
                    break
                removed = S_cur[idx]
                # Temporarily remove
                S_cur[idx] = None
                S_cur_set.discard(removed)
                
                # Recompute sums without the removed element
                sums_bitset2 = np.zeros(MAX_SUM + 1, dtype=bool)
                S_temp = [s for s in S_cur if s is not None]
                n_temp = len(S_temp)
                for i in range(n_temp):
                    for j in range(i, n_temp):
                        for k in range(j, n_temp):
                            s_val = S_temp[i] + S_temp[j] + S_temp[k]
                            if s_val <= MAX_SUM:
                                sums_bitset2[s_val] = True
                
                # Try to find a replacement
                best_replacement = None
                for x in candidates:
                    if x in S_cur_set:
                        continue
                    valid, _ = check_and_get_new_sums(S_temp, sums_bitset2, x)
                    if valid:
                        best_replacement = x
                        break
                
                if best_replacement is not None:
                    S_cur[idx] = best_replacement
                    S_cur_set.add(best_replacement)
                    # Recompute full sums
                    sums_bitset = np.zeros(MAX_SUM + 1, dtype=bool)
                    n_cur = len(S_cur)
                    for i in range(n_cur):
                        for j in range(i, n_cur):
                            for k in range(j, n_cur):
                                s_val = S_cur[i] + S_cur[j] + S_cur[k]
                                if s_val <= MAX_SUM:
                                    sums_bitset[s_val] = True
                    improved = True
                else:
                    # Put back
                    S_cur[idx] = removed
                    S_cur_set.add(removed)
                    sums_bitset = sums_bitset2  # Keep the recomputed version
                
                if time.time() > deadline - 2.0:
                    break
            
            if time.time() > deadline - 2.0:
                break
        
        # Final check
        if len(S_cur) > best_len:
            best_set = S_cur
            best_len = len(S_cur)
    
    return best_set
