import math
import time
import numpy as np
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    n = 30
    H = reduce(math.lcm, range(1, n + 1))
    W = [H // i for i in range(1, n + 1)]
    
    # Try a larger split if time allows: 16/14
    # 3^16 ~ 43M sums. 3^14 ~ 4.7M sums.
    # Memory for 43M int64s is ~344MB.
    # Generation time might be high.
    
    # We will try 15/15 first as it is safe, then if time remains, try 16/14?
    # No, better to commit to one.
    # 15/15 is ~14M sums.
    # 16/14 is ~43M sums.
    
    # Let's stick to 15/15 but optimize the vectorized part?
    # The current code is already vectorized.
    
    # The main issue is the gap.
    # Let's try to use **LLL** as suggested, but simplified.
    # Actually, let's try **Randomized Local Search** with **Simulated Annealing** 
    # starting from the MITM result.
    # Even if steps are large, maybe we can jump over the gap?
    # No, if step is 10^9 and target is 1, we will overshoot.
    
    # Let's try to find a **smaller** subset of weights that can sum to a small number.
    # If we can find a subset of indices $I$ such that $\sum_{i \in I} W_i = 1$, 
    # then we have a solution.
    # This is a subset sum problem.
    # Subset sum with $N=30$ and large weights is hard.
    
    # However, we can use **Meet-in-the-Middle for Subset Sum** to find 
    # if any subset sums to 0, 1, or -1.
    # Current MITM finds the *closest* sum to 0.
    # If the closest sum is 200,000, it means no subset sums to 1.
    
    # Is it possible that the global minimum is indeed ~200,000?
    # For random weights, the minimum non-zero sum is roughly $1/\sqrt{N}$ times the range?
    # No, for random weights, the minimum is very small.
    # But these weights are structured ($L/i$).
    
    # Let's try to **improve the MITM** by using **multiple splits**.
    # Split 1: 0..14 vs 15..29
    # Split 2: 0..13 vs 14..29 (14/16)
    # Split 3: 1..15 vs 0, 16..29 (shifted)
    
    # By trying different splits, we might hit a denser region of the sum space.
    
    def _mitm_search(mid_a, mid_b, W, n, deadline):
        # mid_a is the end index for the first half (exclusive)
        # mid_b is the start index for the second half (inclusive)
        # We assume mid_a == mid_b for contiguous split, but we can allow gaps?
        # No, we must use all indices.
        # So we just change the split point.
        
        sums_a = {0}
        for i in range(mid_a):
            w = W[i]
            new_sums = set()
            for s in sums_a:
                new_sums.add(s + w)
                new_sums.add(s)
                new_sums.add(s - w)
            sums_a = new_sums
            if time.time() > deadline:
                return None
                
        sums_b = {0}
        for i in range(mid_a, n):
            w = W[i]
            new_sums = set()
            for s in sums_b:
                new_sums.add(s + w)
                new_sums.add(s)
                new_sums.add(s - w)
            sums_b = new_sums
            if time.time() > deadline:
                return None
                
        arr_a = np.array(list(sums_a), dtype=np.int64)
        arr_b = np.array(sorted(sums_b), dtype=np.int64)
        
        neg_a = -arr_a
        idx = np.searchsorted(arr_b, neg_a)
        
        idx_right = np.clip(idx, 0, len(arr_b) - 1)
        b_right = arr_b[idx_right]
        idx_left = np.clip(idx - 1, 0, len(arr_b) - 1)
        b_left = arr_b[idx_left]
        
        sum_right = arr_a + b_right
        sum_left = arr_a + b_left
        
        mask_right = sum_right != 0
        mask_left = sum_left != 0
        
        BIG = 10**18
        abs_right = np.abs(sum_right).astype(np.int64)
        abs_left = np.abs(sum_left).astype(np.int64)
        
        abs_right[~mask_right] = BIG
        abs_left[~mask_left] = BIG
        
        min_abs_right = np.min(abs_right)
        min_abs_left = np.min(abs_left)
        
        if min_abs_right < min_abs_left:
            best_val = min_abs_right
            best_idx = np.argmin(abs_right)
            best_a = arr_a[best_idx]
            best_b = b_right[best_idx]
        else:
            best_val = min_abs_left
            best_idx = np.argmin(abs_left)
            best_a = arr_a[best_idx]
            best_b = b_left[best_idx]
            
        if best_val >= BIG:
            return None
            
        coeffs_a = _find_coeffs(int(best_a), 0, mid_a, W)
        coeffs_b = _find_coeffs(int(best_b), mid_a, n, W)
        
        if coeffs_a is not None and coeffs_b is not None:
            final_coeffs = coeffs_a + coeffs_b
            total = sum(c * W[i] for i, c in enumerate(final_coeffs))
            if total != 0:
                return final_coeffs, int(best_val)
                
        return None

    best_coeffs = None
    best_val = float('inf')
    
    # Try split 15
    res = _mitm_search(15, 15, W, n, deadline)
    if res:
        best_coeffs, best_val = res
        
    # Try split 14 if time permits
    if time.time() < deadline - 20:
        res = _mitm_search(14, 14, W, n, deadline)
        if res and res[1] < best_val:
            best_coeffs, best_val = res
            
    # Try split 16 if time permits
    if time.time() < deadline - 20:
        res = _mitm_search(16, 16, W, n, deadline)
        if res and res[1] < best_val:
            best_coeffs, best_val = res
            
    if best_coeffs is None:
        return _fallback(W, n)
        
    return best_coeffs

def _find_coeffs(target, start, end, W):
    if start == end:
        return [] if target == 0 else None
        
    max_rem = sum(W[start:end])
    if abs(target) > max_rem:
        return None
        
    w = W[start]
    for c in [-1, 0, 1]:
        res = _find_coeffs(target - c * w, start + 1, end, W)
        if res is not None:
            return [c] + res
    return None

def _fallback(W, n):
    coeffs = [0] * n
    curr = 0
    for i in range(n - 1, -1, -1):
        w = W[i]
        best_c = 0
        best_val = curr
        for c in [-1, 0, 1]:
            val = curr + c * w
            if abs(val) < abs(best_val):
                best_val = val
                best_c = c
        coeffs[i] = best_c
        curr = best_val
    
    if curr == 0:
        min_idx = min(range(n), key=lambda i: W[i])
        coeffs[min_idx] = 1
        curr = sum(c * W[i] for i, c in enumerate(coeffs))
        
    return coeffs
