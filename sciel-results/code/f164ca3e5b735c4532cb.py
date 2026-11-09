import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    # Sieve primes up to 10^6
    limit = 1000000
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.array([i for i, val in enumerate(is_prime) if val])

    best_k = 1
    best_s = 1
    left_idx = 1  # Global index of left pointer

    # last_occurrence maps phi_value -> last global index
    # We will prune this to keep only indices within the last 200 steps
    last_occurrence = {}

    def update_window(idx, val):
        nonlocal best_k, best_s, left_idx
        
        # Check if this phi value has been seen recently
        if val in last_occurrence:
            prev_idx = last_occurrence[val]
            if prev_idx + 1 > left_idx:
                left_idx = prev_idx + 1
        
        # Update the last occurrence
        last_occurrence[val] = idx
        
        # Update best if current window is longer
        current_len = idx - left_idx + 1
        if current_len > best_k:
            best_k = current_len
            best_s = left_idx

        # Prune old entries: if index is too far back, it can't affect a window of size <= 200
        # We can do this periodically or every step. Every step is cheap if dict is small.
        # Threshold: idx - 200
        threshold = idx - 200
        # Since dict is small, we can just iterate and delete
        # But doing this every step might be overhead. 
        # Let's do it every 1000 steps or when dict size > 1000.
        # Actually, the dict size is bounded by 200 distinct values in the last 200 indices.
        # So the dict size will never exceed ~200. 
        # So we can just prune every step? No, iterating over 200 items 10^12 times is too slow.
        # We should prune less frequently.
        # Let's prune when len(last_occurrence) > 1000. 
        # But wait, if we don't prune, the dict grows. 
        # The values in last_occurrence are indices. 
        # If we don't prune, the dict will have 10^12 entries. That's bad.
        # We MUST prune.
        # How to prune efficiently?
        # We can keep a list of (phi, idx) and periodically filter?
        # Or, we can just delete keys where value < idx - 200.
        # Since the dict is small (max 200 entries if we prune aggressively enough), 
        # we can just do:
        # for k in list(last_occurrence.keys()):
        #     if last_occurrence[k] < idx - 200:
        #         del last_occurrence[k]
        # This is O(200) per step. 10^12 * 200 is still too slow.
        
        # Alternative: Don't prune every step. Prune every 10000 steps.
        # The dict size will grow to ~10000 * (density of distinct phis).
        # In 10000 numbers, there are ~10000 distinct phis.
        # So dict size ~10000.
        # Pruning 10000 items every 10000 steps is O(1) amortized.
        # This is acceptable.

    # Main loop
    segment_size = 1000000
    start_val = 1
    step_count = 0

    while start_val <= 10**12:
        end_val = min(start_val + segment_size - 1, 10**12)
        n = end_val - start_val + 1

        # Segmented Sieve for phi
        vals = np.arange(start_val, end_val + 1, dtype=np.int64)
        phis = np.arange(start_val, end_val + 1, dtype=np.int64)

        for p in primes:
            if p * p > end_val:
                break
            first_mult = ((start_val + p - 1) // p) * p
            start_idx = first_mult - start_val
            step = p
            mask_indices = np.arange(start_idx, n, step)

            if len(mask_indices) == 0:
                continue

            phi_slice = phis[mask_indices]
            phis[mask_indices] = phi_slice // p * (p - 1)

            v_slice = vals[mask_indices]
            # Remove all factors of p from v_slice
            # Loop while any element is divisible by p
            # This is a tight loop, but p is small for most iterations
            while True:
                mask_div = v_slice % p == 0
                if not np.any(mask_div):
                    break
                v_slice[mask_div] //= p
            vals[mask_indices] = v_slice

        mask_rem = vals > 1
        if np.any(mask_rem):
            phis[mask_rem] = phis[mask_rem] // vals[mask_rem] * (vals[mask_rem] - 1)

        # Update window for each phi value
        # We do this in a Python loop.
        # To speed up pruning, we'll prune every 10000 iterations.
        for i in range(n):
            idx = start_val + i
            val = phis[i]
            update_window(idx, val)
            step_count += 1
            if step_count % 10000 == 0:
                # Prune last_occurrence
                threshold = idx - 200
                keys_to_remove = [k for k, v in last_occurrence.items() if v < threshold]
                for k in keys_to_remove:
                    del last_occurrence[k]

        start_val = end_val + 1
        if time.time() > deadline:
            break

    # Guarantee valid result
    if best_s + best_k - 1 > 10**12:
        best_s = 1
        best_k = 1

    return {'start': int(best_s), 'length': int(best_k)}
