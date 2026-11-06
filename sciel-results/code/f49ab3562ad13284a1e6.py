# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_result = []
    best_len = 0
    
    MAX_SUM = 3 * 4096 + 1
    
    def is_valid_addition(S, x, sums_arr):
        """Check if adding x to S keeps all triple sums unique."""
        n = len(S)
        
        # (x, x, x)
        s = 3 * x
        if sums_arr[s]:
            return False
        
        # (x, x, y) for y in S
        for y in S:
            s = 2 * x + y
            if sums_arr[s]:
                return False
        
        # (x, y, z) for y <= z in S
        for i in range(n):
            y = S[i]
            for j in range(i, n):
                z = S[j]
                s = x + y + z
                if sums_arr[s]:
                    return False
        
        return True
    
    def add_element(S, sums_arr, x):
        """Add x to S and update sums_arr with all new triple sums."""
        n = len(S)
        S.append(x)
        sums_arr[3 * x] = 1
        for i in range(n):
            y = S[i]
            sums_arr[2 * x + y] = 1
            for j in range(i, n):
                z = S[j]
                sums_arr[x + y + z] = 1
        
    def greedy_with_order(candidates, max_elements=80):
        """Run greedy construction with a given candidate order."""
        S = []
        sums_arr = bytearray(MAX_SUM)
        
        for x in candidates:
            if len(S) >= max_elements:
                break
            if time.time() > deadline:
                break
            
            if is_valid_addition(S, x, sums_arr):
                add_element(S, sums_arr, x)
        
        return S

    # Strategy 1: Exponential spacing
    exp_candidates = []
    val = 1
    while val <= 4096:
        exp_candidates.append(val)
        val = int(val * 1.5) + 1
    for v in range(2048, 4097, 100):
        if v not in exp_candidates:
            exp_candidates.append(v)
    exp_candidates.sort()
    
    result = greedy_with_order(exp_candidates)
    if len(result) > best_len:
        best_len = len(result)
        best_result = result
        
    # Strategy 2: Base spacing
    def construct_with_base(base_step=64):
        base_elements = list(range(1, 4097, base_step))
        S = []
        sums_arr = bytearray(MAX_SUM)
        
        for x in base_elements:
            if len(S) >= 80:
                break
            if time.time() > deadline:
                break
            if is_valid_addition(S, x, sums_arr):
                add_element(S, sums_arr, x)
        
        if len(S) >= 80:
            return S
        
        S_sorted = sorted(S)
        candidates_to_try = []
        for i in range(len(S_sorted) - 1):
            lo = S_sorted[i]
            hi = S_sorted[i + 1]
            for frac in [0.25, 0.5, 0.75]:
                mid = int(lo + frac * (hi - lo))
                if 1 <= mid <= 4096 and mid not in S:
                    candidates_to_try.append(mid)
        
        for v in range(1, min(200, S_sorted[0] if S_sorted else 1)):
            if v not in S:
                candidates_to_try.append(v)
        
        random.Random(42).shuffle(candidates_to_try)
        
        for x in candidates_to_try:
            if len(S) >= 80:
                break
            if time.time() > deadline:
                break
            if is_valid_addition(S, x, sums_arr):
                add_element(S, sums_arr, x)
        
        return S

    for step in [32, 64, 128, 256]:
        if time.time() > deadline - 2:
            break
        result = construct_with_base(step)
        if len(result) > best_len:
            best_len = len(result)
            best_result = result

    # Strategy 3: Randomized greedy with multiple restarts
    # Increased restarts due to faster check
    if time.time() < deadline - 5:
        for seed in range(1, 100):
            if time.time() > deadline - 2:
                break
            rng = random.Random(seed)
            candidates = list(range(1, 4097))
            rng.shuffle(candidates)
            
            S = []
            sums_arr = bytearray(MAX_SUM)
            
            for x in candidates:
                if len(S) >= 80:
                    break
                if time.time() > deadline - 2:
                    break
                
                if is_valid_addition(S, x, sums_arr):
                    add_element(S, sums_arr, x)
            
            if len(S) > best_len:
                best_len = len(S)
                best_result = S
                
                # If we hit 80, we can stop early? No, we want to maximize, but 80 is the cap.
                # If we hit 80, we can break out of restarts?
                # The task says "Maximize cardinality". 80 is the max allowed.
                # If we find 80, we can return immediately.
                if best_len == 80:
                    return best_result

    # Strategy 4: Hybrid
    if time.time() < deadline - 3:
        for seed in range(1, 30):
            if time.time() > deadline - 1:
                break
            rng = random.Random(seed + 100)
            
            exp_part = []
            val = 1
            while val <= 4096 and len(exp_part) < 30:
                exp_part.append(val)
                val = int(val * 1.4) + 1
            
            rand_part = list(range(1, 4097))
            rng.shuffle(rand_part)
            
            candidates = exp_part + rand_part
            seen = set()
            unique_candidates = []
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    unique_candidates.append(c)
            
            result = greedy_with_order(unique_candidates)
            if len(result) > best_len:
                best_len = len(result)
                best_result = result
                if best_len == 80:
                    return best_result

    return best_result if best_result else [1, 2, 4]
# EVOLVE-BLOCK-END
