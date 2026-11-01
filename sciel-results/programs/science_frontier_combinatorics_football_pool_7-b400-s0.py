import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin for 120s limit (evaluator kills at 150s, but task says 120s)
    
    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute place values for index conversion
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q
        
    # Precompute all words as indices
    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q
            
    # Precompute ball_indices: for each word, list of indices within Hamming distance 1
    ball_indices = [[] for _ in range(total)]
    for i in range(total):
        w = words[i]
        ball_indices[i].append(i)
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball_indices[i].append(new_idx)
                
    # Convert to numpy arrays for faster vectorized operations
    ball_indices_np = [np.array(b, dtype=np.int32) for b in ball_indices]
    
    # Precompute a membership matrix for quick intersection checks?
    # membership[i, j] = 1 if j is in ball(i)
    # This is 2187x2187 boolean matrix. 4.7M booleans ~ 4.7MB. Acceptable.
    # But we can also just use sets for the small balls if needed.
    # For swap validity, we need: for all x in ball(c_remove) \ ball(c_add), cnt[x] >= 2.
    # Using numpy:
    # mask_remove = np.zeros(total, dtype=bool); mask_remove[ball_indices_np[c_remove]] = True
    # mask_add = np.zeros(total, dtype=bool); mask_add[ball_indices_np[c_add]] = True
    # candidates = mask_remove & ~mask_add
    # if np.any(cnt[candidates] < 2): invalid
    
    # Greedy construction
    def greedy_build():
        covered = np.zeros(total, dtype=bool)
        code = []
        
        while not np.all(covered):
            best_c = -1
            max_new = -1
            for c in range(total):
                ball = ball_indices_np[c]
                new_cov = np.sum(~covered[ball])
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == ball_size:
                        break
            code.append(best_c)
            covered[ball_indices_np[best_c]] = True
        return code
    
    # Initial greedy
    current_code = set(greedy_build())
    best_code = list(current_code)
    best_size = len(current_code)
    
    # Initialize counts
    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        cnt[ball_indices_np[c]] += 1
        
    def run_delete_phase(code_set, cnt_arr):
        """Removes redundant codewords. Returns True if any removed."""
        changed = False
        # Iterate over a copy to allow modification
        for c in list(code_set):
            ball = ball_indices_np[c]
            if np.min(cnt_arr[ball]) >= 2:
                code_set.remove(c)
                cnt_arr[ball] -= 1
                changed = True
        return changed
    
    def try_swap(c_remove, c_add, code_set, cnt_arr):
        """Try a swap. Returns True if successful."""
        if c_add in code_set:
            return False
        if c_remove == c_add:
            return False
            
        ball_rem = ball_indices_np[c_remove]
        ball_add = ball_indices_np[c_add]
        
        # We need to check if removing c_remove and adding c_add keeps coverage.
        # cnt' = cnt - ball_rem + ball_add
        # We need cnt' > 0 everywhere.
        # This is equivalent to:
        # For x in ball_rem: cnt[x] - 1 + (1 if x in ball_add else 0) > 0
        # For x not in ball_rem: cnt[x] + (1 if x in ball_add else 0) > 0 (always true since cnt[x] >= 1)
        
        # So we only need to check x in ball_rem.
        # Specifically, for x in ball_rem \ ball_add, we need cnt[x] >= 2.
        # For x in ball_rem ∩ ball_add, cnt'[x] = cnt[x], which is >= 1.
        
        # Efficient check:
        # Create a mask for ball_add
        # Or just iterate ball_rem and check membership in ball_add set?
        # Ball size is 15. Iterating 15 elements is fast.
        # But let's use numpy for vectorization if possible.
        
        # Simple loop might be faster due to overhead of numpy array creation for small sizes
        # But let's try a hybrid:
        # in_add = np.zeros(total, dtype=bool)
        # in_add[ball_add] = True
        # mask = in_add[ball_rem]
        # # For x in ball_rem where NOT in_add, check cnt[x] >= 2
        
        # Actually, creating a mask of size 2187 every swap is expensive.
        # Let's use the property that balls are small.
        set_add = set(ball_add.tolist())
        
        valid = True
        for x in ball_rem:
            if x not in set_add:
                if cnt_arr[x] < 2:
                    valid = False
                    break
                    
        if not valid:
            return False
            
        # Perform swap
        code_set.remove(c_remove)
        code_set.add(c_add)
        cnt_arr[ball_rem] -= 1
        cnt_arr[ball_add] += 1
        return True

    # Local Search Loop
    while time.time() < deadline - 5.0:
        # 1. Delete Phase
        while run_delete_phase(current_code, cnt):
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                
        # 2. Swap Phase
        swaps_attempted = 0
        max_swaps = 500
        while swaps_attempted < max_swaps and time.time() < deadline - 5.0:
            swaps_attempted += 1
            if len(current_code) == 0:
                break
                
            # Pick random codeword to remove
            c = random.choice(list(current_code))
            
            # Pick random candidate to add
            c_prime = random.randint(0, total - 1)
            
            # Try the swap
            if try_swap(c, c_prime, current_code, cnt):
                # After a valid swap, run delete phase
                while run_delete_phase(current_code, cnt):
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                    
                    # Early exit if we've hit a very good solution
                    if best_size <= 160:
                        break
                        
                if best_size <= 160:
                    break
                    
        if best_size <= 160:
            break

    # Final result conversion
    result = []
    for idx in best_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
        
    return result
