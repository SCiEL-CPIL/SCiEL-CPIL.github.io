import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0 # Leave 10s margin
    
    # 1. Sieve of Eratosthenes up to 10^6
    limit = 1000000
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # 2. Function to compute phi for a segment [L, L+W)
    def compute_phi_segment(L, W):
        phi_vals = np.arange(L, L + W, dtype=np.int64)
        max_prime_sq = L + W
        for p in primes:
            if p * p > max_prime_sq:
                break
            first_multiple = L + (p - L % p) % p
            if first_multiple >= L + W:
                continue
            indices = np.arange(first_multiple - L, W, p)
            phi_vals[indices] -= phi_vals[indices] // p
        return phi_vals

    best_k = 1
    best_s = 1
    
    W = 1000000 # 10^6
    max_L = 100000000 # 10^8
    
    buffer_len = 200
    prev_phis = []
    
    current_L = 1
    while current_L <= max_L:
        if time.time() - start_time > deadline:
            break
            
        end_L = min(current_L + W, max_L + 1)
        seg_W = end_L - current_L
        
        phi_vals = compute_phi_segment(current_L, seg_W)
        
        if prev_phis:
            combined = np.concatenate([np.array(prev_phis[-buffer_len:]), phi_vals])
            start_idx_offset = -len(prev_phis[-buffer_len:])
        else:
            combined = phi_vals
            start_idx_offset = 0
            
        # Sliding window
        last_seen = {}
        left = 0
        max_len_in_seg = 0
        max_start_idx = 0
        
        # Use local variables for speed
        ls = last_seen
        for right in range(len(combined)):
            val = combined[right]
            if val in ls:
                if ls[val] >= left:
                    left = ls[val] + 1
            ls[val] = right
            
            current_len = right - left + 1
            if current_len > max_len_in_seg:
                max_len_in_seg = current_len
                max_start_idx = left
                
        if max_len_in_seg > best_k:
            actual_start = current_L + start_idx_offset + max_start_idx
            if actual_start >= 1:
                best_k = max_len_in_seg
                best_s = actual_start
        
        prev_phis = list(phi_vals[-buffer_len:])
        current_L = end_L
        
    return {'start': best_s, 'length': best_k}
