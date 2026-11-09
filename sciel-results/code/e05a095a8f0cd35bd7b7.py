import numpy as np
import time

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
    
    # Sliding window state
    # We process segments. We need to carry over the tail of the previous segment.
    # Let's keep a buffer of the last 'buffer_size' phi values and their global indices.
    # buffer_size = 1000 (safe upper bound for k)
    buffer_size = 1000
    phi_buffer = [] # list of (global_index, phi_value)
    
    # Sliding window variables
    left_idx = 0 # Global index of left pointer
    # We need to know the last seen index for each phi value in the current window.
    # Since we process sequentially, we can maintain a dict `last_seen` for values in the window.
    # But the window moves.
    
    # Actually, simpler:
    # Keep a list `recent_phis` of the last `buffer_size` phi values.
    # For each new phi, check against `recent_phis` to find the earliest collision.
    # If collision at offset `d` (i.e., `recent_phis[-d-1] == new_phi`), then left moves to `current_idx - d`.
    # This is O(K) per step. If K is small (e.g., < 50), it's very fast.
    
    recent_phis = []
    current_idx = 0
    
    def update_window(idx, val):
        nonlocal best_k, best_s, recent_phis, left_idx
        
        # Find collision in recent_phis
        # recent_phis contains values from indices [idx - len(recent_phis), ..., idx-1]
        # We want to find the largest j such that recent_phis[j] == val
        # If found at position `pos` from end (0-based, 0 is last), collision index is `idx - 1 - pos`.
        # Then left_idx must be > collision index.
        
        collision_idx = -1
        # Search backwards
        for i in range(len(recent_phis) - 1, -1, -1):
            if recent_phis[i] == val:
                collision_idx = idx - 1 - (len(recent_phis) - 1 - i) # This is getting complicated.
                # Simpler:
                # recent_phis[0] is idx - len(recent_phis)
                # recent_phis[i] is idx - len(recent_phis) + i
                # If recent_phis[i] == val, collision at global index `idx - len(recent_phis) + i`.
                break
        
        if collision_idx != -1:
            # The value `val` was seen at `collision_idx`.
            # So the new left must be at least `collision_idx + 1`.
            if collision_idx + 1 > left_idx:
                left_idx = collision_idx + 1
        
        # Update best
        current_len = idx - left_idx + 1
        if current_len > best_k:
            best_k = current_len
            best_s = left_idx
            
        # Add to buffer
        recent_phis.append(val)
        if len(recent_phis) > buffer_size:
            recent_phis.pop(0)
            
    # Main loop
    segment_size = 1000000
    start_val = 1
    while start_val <= 10**12:
        end_val = min(start_val + segment_size - 1, 10**12)
        
        # Segmented Sieve for [start_val, end_val]
        n = end_val - start_val + 1
        vals = np.arange(start_val, end_val + 1)
        phis = np.arange(start_val, end_val + 1)
        
        for p in primes:
            if p * p > end_val:
                break
            # Find first multiple of p in [start_val, end_val]
            first_mult = ((start_val + p - 1) // p) * p
            if first_mult < p * p:
                first_mult = p * p # Optimization? No, we need to handle all multiples.
                # Actually, for phi, we need to divide out p from all multiples.
                
            # Indices in the array
            start_idx = first_mult - start_val
            step = p
            
            # Vectorized update
            # vals[start_idx::step]
            # phis[start_idx::step]
            
            mask_indices = np.arange(start_idx, n, step)
            
            # Check if divisible (should be, by construction)
            # vals[mask_indices] is divisible by p
            # phis[mask_indices] = phis[mask_indices] // p * (p-1)
            # vals[mask_indices] //= p (need to divide out all factors of p? No, just once for the formula, but we need to track if p is fully removed?
            # The standard algorithm:
            # for each p:
            #   for each multiple m of p:
            #     while m % p == 0: m /= p
            #     phi[m] = phi[m] / p * (p-1)
            
            # Vectorized "while m % p == 0" is hard.
            # But we can do:
            # tmp = vals[mask_indices]
            # while any(tmp % p == 0):
            #   mask2 = tmp % p == 0
            #   tmp[mask2] //= p
            #   # update phi? No, phi update happens once per prime factor.
            
            # Actually, the formula $\phi(n) = n \prod (1-1/p)$ works if we apply it for each distinct prime factor.
            # So we just need to know if p is a factor.
            # If we divide `vals` by p as much as possible, we ensure we don't process p again for the same number.
            # But for the phi calculation, we just do `phi = phi // p * (p-1)` once.
            
            # So:
            phi_slice = phis[mask_indices]
            phi_slice = phi_slice // p * (p - 1)
            phis[mask_indices] = phi_slice
            
            # Remove p from vals
            v_slice = vals[mask_indices]
            # Divide by p as long as divisible
            # This loop might be slow if done in Python.
            # But the number of iterations is small (log_p n).
            # We can do:
            for _ in range(10): # Max power of 2 is ~40 for 10^12. 10 is not enough.
                 mask_div = v_slice % p == 0
                 if not np.any(mask_div): break
                 v_slice[mask_div] //= p
            vals[mask_indices] = v_slice
            
        # Handle remaining prime factors in vals
        # If vals[i] > 1, it is a prime factor.
        mask_rem = vals > 1
        if np.any(mask_rem):
            # phis[mask_rem] = phis[mask_rem] // vals[mask_rem] * (vals[mask_rem] - 1)
            phis[mask_rem] = phis[mask_rem] // vals[mask_rem] * (vals[mask_rem] - 1)
            
        # Update sliding window
        for i in range(n):
            if time.time() > deadline:
                break
            idx = start_val + i
            val = int(phis[i])
            update_window(idx, val)
            
        start_val = end_val + 1
        if time.time() > deadline:
            break
            
    return {'start': best_s, 'length': best_k}
