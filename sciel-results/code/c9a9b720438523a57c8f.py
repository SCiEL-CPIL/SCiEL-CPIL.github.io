import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 120s limit
    
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
                
    # Convert to numpy arrays for faster access
    ball_indices_np = [np.array(b, dtype=np.int32) for b in ball_indices]
    
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
    best_size = len(best_code)
    
    # Initialize counts
    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        cnt[ball_indices_np[c]] += 1
        
    def run_delete_phase(code_set, cnt_arr):
        """Removes redundant codewords. Returns True if any removed."""
        changed = False
        for c in list(code_set):
            ball = ball_indices_np[c]
            if np.min(cnt_arr[ball]) >= 2:
                code_set.remove(c)
                cnt_arr[ball] -= 1
                changed = True
        return changed
    
    def try_swap(c_remove, c_add, code_set, cnt_arr):
        """Try a swap. Returns True if successful."""
        ball_rem = ball_indices_np[c_remove]
        ball_add = ball_indices_np[c_add]
        
        # Check if swap is valid
        # Points that lose coverage: ball_rem \ ball_add
        # Points that gain coverage: ball_add \ ball_rem
        # A swap is valid if for all x in ball_rem, cnt[x] - 1 + (1 if x in ball_add else 0) > 0
        # i.e., cnt[x] > 0 for x not in ball_add, and cnt[x] >= 1 for x in ball_add (always true since cnt[x] >= 1 before swap)
        
        # More precisely: after swap, cnt'[x] = cnt[x] - 1_{x in ball_rem} + 1_{x in ball_add}
        # We need cnt'[x] > 0 for all x.
        # For x not in ball_rem and not in ball_add: cnt'[x] = cnt[x] > 0 (already valid)
        # For x in ball_rem and not in ball_add: cnt'[x] = cnt[x] - 1, need cnt[x] >= 2
        # For x in ball_add and not in ball_rem: cnt'[x] = cnt[x] + 1 > 0 (always true)
        # For x in both: cnt'[x] = cnt[x], need cnt[x] > 0 (always true)
        
        # So we only need to check: for x in ball_rem \ ball_add, cnt[x] >= 2
        
        in_rem_not_add = [x for x in ball_rem if x not in set(ball_add)]
        if in_rem_not_add:
            if np.min(cnt_arr[in_rem_not_add]) < 2:
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
            
            if c_prime == c:
                continue
            if c_prime in current_code:
                continue
                
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
