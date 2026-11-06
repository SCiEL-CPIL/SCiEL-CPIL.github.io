import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 4096
    max_sum = 3 * n
    
    best_set = []
    best_len = 0
    
    # Initial guess: Power of 3s is a valid B_3[1] set
    power3_set = [3**i for i in range(8) if 3**i <= n]
    best_set = list(power3_set)
    best_len = len(power3_set)
    
    def is_valid_set(S):
        if not S:
            return True
        sums_used = bytearray(max_sum + 1)
        S = list(S)
        k = len(S)
        for i in range(k):
            si = S[i]
            for j in range(i, k):
                sj = S[j]
                for l in range(j, k):
                    sl = S[l]
                    s = si + sj + sl
                    if sums_used[s]:
                        return False
                    sums_used[s] = 1
        return True

    def can_add_fast(x, S, sums_used):
        """
        Optimized check if x can be added to S.
        """
        k = len(S)
        
        # 1. Check 3x
        if sums_used[3 * x]:
            return False, None
        
        # 2. Check 2x + y for y in S
        sums_2x = []
        for y in S:
            s = 2 * x + y
            if sums_used[s]:
                return False, None
            sums_2x.append(s)
            
        # 3. Check x + y + z for y, z in S
        sums_x = []
        for i in range(k):
            yi = S[i]
            for j in range(i, k):
                zj = S[j]
                s = x + yi + zj
                if sums_used[s]:
                    return False, None
                sums_x.append(s)
                
        # 4. Check internal collisions between sums_2x and sums_x
        # 2x + y1 == x + y2 + z3  =>  x + y1 == y2 + z3
        # Since S is a B_3[1] set, it is also a B_2[1] set (pair sums unique).
        # So y2 + z3 is unique for each pair.
        # We just need to check if any value in sums_2x appears in sums_x.
        
        # Optimization: sums_2x size is k, sums_x size is k(k+1)/2.
        # Convert sums_2x to set for O(1) lookup.
        set_2x = set(sums_2x)
        for s in sums_x:
            if s in set_2x:
                return False, None
                
        # All checks passed
        new_sums = sums_2x + sums_x + [3 * x]
        return True, new_sums

    def greedy_build(ordering, sums_used):
        S = []
        for x in ordering:
            valid, new_sums = can_add_fast(x, S, sums_used)
            if valid:
                S.append(x)
                for s in new_sums:
                    sums_used[s] = 1
        return S

    # Strategy 1: Randomized Greedy with multiple seeds
    # Increased restarts due to faster can_add
    
    num_restarts = 300
    
    for seed in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        
        # Strategy: Mix of random and biased
        # 50% pure random, 50% biased towards smaller numbers
        if seed % 2 == 0:
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
        else:
            # Bias: Sort by (random.random(), value)
            # This tends to pick smaller numbers if random is small?
            # Actually, let's just shuffle and then sort by a random key?
            # Simpler: Shuffle, but ensure first few are small?
            # Let's just use random shuffle for now, the speedup allows more runs.
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
            
        sums_used = bytearray(max_sum + 1)
        S = greedy_build(candidates, sums_used)
        
        if len(S) > best_len:
            best_set = S
            best_len = len(S)

    # Strategy 2: Local Search - Add if possible
    if time.time() < deadline - 2:
        current_S = list(best_set)
        sums_used = bytearray(max_sum + 1)
        for i in range(len(current_S)):
            si = current_S[i]
            for j in range(i, len(current_S)):
                sj = current_S[j]
                for k in range(j, len(current_S)):
                    sk = current_S[k]
                    sums_used[si + sj + sk] = 1
        
        # Try to add any element
        for x in range(1, n + 1):
            if x in current_S:
                continue
            valid, new_sums = can_add_fast(x, current_S, sums_used)
            if valid:
                current_S.append(x)
                for s in new_sums:
                    sums_used[s] = 1
                if len(current_S) > best_len:
                    best_set = current_S
                    best_len = len(current_S)
                    # If we improved, restart the scan? 
                    # No, just continue.
        
    # Final validation
    if best_set and not is_valid_set(best_set):
        best_set = [3**i for i in range(8) if 3**i <= n]

    return best_set
