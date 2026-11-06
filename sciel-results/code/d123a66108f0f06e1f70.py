# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_result = []
    best_len = 0
    
    MAX_SUM = 3 * 4096 + 1

    def is_valid_addition_np(S, x, sums_arr):
        """Check if adding x to S keeps all triple sums unique using NumPy."""
        n = len(S)
        if n == 0:
            return True
        
        S_arr = np.array(S, dtype=np.int64)
        
        # Generate all new triple sums involving x
        # (x, x, x)
        # (x, x, y) for y in S
        # (x, y, z) for y <= z in S
        new_sums = [3 * x]
        for y in S:
            new_sums.append(2 * x + y)
        for i in range(n):
            for j in range(i, n):
                new_sums.append(x + S[i] + S[j])
        
        new_sums_arr = np.array(new_sums, dtype=np.int64)
        
        # Check for collisions with existing sums
        existing_mask = sums_arr[new_sums_arr]
        if np.any(existing_mask):
            return False
        
        # Check for internal collisions in new_sums
        if len(new_sums) > 1:
            sorted_new = np.sort(new_sums_arr)
            if np.any(np.diff(sorted_new) == 0):
                return False
        
        return True

    def add_element_np(S, sums_arr, x):
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

    def greedy_with_order(candidates, max_elements=80, deadline_check=True):
        """Run greedy construction with a given candidate order."""
        S = []
        sums_arr = bytearray(MAX_SUM)
        
        for x in candidates:
            if len(S) >= max_elements:
                break
            if deadline_check and time.time() > deadline:
                break
            
            if is_valid_addition_np(S, x, sums_arr):
                add_element_np(S, sums_arr, x)
        
        return S

    def local_search(S, sums_arr, max_iterations=200):
        """Try to improve S by swapping elements."""
        S = list(S)
        n = len(S)
        
        for _ in range(max_iterations):
            if time.time() > deadline - 2:
                break
            
            improved = False
            rng = random.Random()
            
            # Try removing one element and adding a better one
            for i in range(n):
                if time.time() > deadline - 2:
                    break
                
                # Remove S[i]
                removed = S[i]
                S[i] = S[-1]
                S.pop()
                
                # Rebuild sums_arr without removed
                new_sums_arr = bytearray(MAX_SUM)
                for a in range(len(S)):
                    for b in range(a, len(S)):
                        for c in range(b, len(S)):
                            s = S[a] + S[b] + S[c]
                            new_sums_arr[s] = 1
                
                # Try to add a new element
                candidates = list(range(1, 4097))
                random.Random().shuffle(candidates)
                
                found = False
                for x in candidates:
                    if x == removed or x in S:
                        continue
                    if is_valid_addition_np(S, x, new_sums_arr):
                        S.append(x)
                        S.sort()
                        found = True
                        break
                
                if not found:
                    # Restore
                    S.append(removed)
                    S.sort()
                    sums_arr = bytearray(MAX_SUM)
                    for a in range(len(S)):
                        for b in range(a, len(S)):
                            for c in range(b, len(S)):
                                s = S[a] + S[b] + S[c]
                                sums_arr[s] = 1
                else:
                    improved = True
                    sums_arr = new_sums_arr
                    break
            
            if not improved:
                break
        
        return S, sums_arr

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
            if is_valid_addition_np(S, x, sums_arr):
                add_element_np(S, sums_arr, x)
        
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
            if is_valid_addition_np(S, x, sums_arr):
                add_element_np(S, sums_arr, x)
        
        return S

    for step in [32, 64, 128, 256]:
        if time.time() > deadline - 2:
            break
        result = construct_with_base(step)
        if len(result) > best_len:
            best_len = len(result)
            best_result = result

    # Strategy 3: Randomized greedy with multiple restarts
    if time.time() < deadline - 5:
        for seed in range(1, 200):
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
                
                if is_valid_addition_np(S, x, sums_arr):
                    add_element_np(S, sums_arr, x)
            
            if len(S) > best_len:
                best_len = len(S)
                best_result = S
                
                if best_len == 80:
                    return best_result

    # Strategy 4: Hybrid
    if time.time() < deadline - 3:
        for seed in range(1, 50):
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
