import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 110.0

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

    # Dictionary to track last occurrence of each phi value
    # Maps phi_value -> last global index where this phi was seen
    last_occurrence = {}
    left_idx = 0  # Global index of left pointer of current valid window

    def update_window(idx, val):
        nonlocal best_k, best_s, left_idx

        # Check if this phi value has been seen before
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

    # Main loop with larger segments
    segment_size = 1000000
    start_val = 1
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

            phi_slice = phis[mask_indices]
            phis[mask_indices] = phi_slice // p * (p - 1)

            v_slice = vals[mask_indices]
            for _ in range(40):
                mask_div = v_slice % p == 0
                if not np.any(mask_div):
                    break
                v_slice[mask_div] //= p
            vals[mask_indices] = v_slice

        mask_rem = vals > 1
        if np.any(mask_rem):
            phis[mask_rem] = phis[mask_rem] // vals[mask_rem] * (vals[mask_rem] - 1)

        # Update window for each phi value
        phis_list = phis.tolist()
        for i, val in enumerate(phis_list):
            idx = start_val + i
            update_window(idx, val)

        # Clean up old entries from last_occurrence to save memory
        # Only keep entries with index >= left_idx
        if len(last_occurrence) > 1000000:
            keys_to_remove = [k for k, v in last_occurrence.items() if v < left_idx]
            for k in keys_to_remove:
                del last_occurrence[k]

        start_val = end_val + 1
        if time.time() > deadline:
            break

    # Guarantee valid result
    if best_s + best_k - 1 > 10**12:
        best_s = 1
        best_k = 1

    return {'start': best_s, 'length': best_k}
