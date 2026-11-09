import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin before 120s kill
    
    # Generate primes up to 10^6 using sieve
    limit = 10**6
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    primes = np.array(np.nonzero(sieve)[0])
    primes = primes[primes >= 2]
    
    # Segmented phi computation
    def compute_phi_segmented(L, R):
        """Compute phi(n) for n in [L, R]."""
        n_count = R - L + 1
        phi = np.arange(L, R + 1, dtype=np.int64)
        remaining = np.arange(L, R + 1, dtype=np.int64)
        
        for p in primes:
            p = int(p)
            if p * p > R:
                break
            first = ((L + p - 1) // p) * p
            if first > R:
                continue
            # Get indices of multiples of p
            idxs = np.arange(first - L, n_count, p)
            # Check which still have p as a factor
            mask = remaining[idxs] % p == 0
            if not np.any(mask):
                continue
            selected = idxs[mask]
            phi[selected] = (phi[selected] // p) * (p - 1)
            # Remove all factors of p from remaining
            rem_vals = remaining[selected].copy()
            # Repeatedly divide by p
            while True:
                mask2 = rem_vals % p == 0
                if not np.any(mask2):
                    break
                rem_vals[mask2] //= p
            remaining[selected] = rem_vals
        
        # Handle remaining prime factors
        mask = remaining > 1
        if np.any(mask):
            phi[mask] = (phi[mask] // remaining[mask]) * (remaining[mask] - 1)
        
        return phi
    
    # Find best run using sliding window
    def find_best_run(phi_arr, offset):
        n = len(phi_arr)
        best_len = 0
        best_idx = 0
        seen = set()
        start = 0
        
        for end in range(n):
            val = phi_arr[end]
            if val in seen:
                while phi_arr[start] != val:
                    seen.remove(phi_arr[start])
                    start += 1
                start += 1
            else:
                seen.add(val)
            current_len = end - start + 1
            if current_len > best_len:
                best_len = current_len
                best_idx = start
        return best_len, best_idx + offset
    
    best_k = 1
    best_start = 1
    
    seg_size = 500000
    overlap = 200
    stride = seg_size - overlap
    
    current_L = 1
    while current_L + seg_size - 1 <= 10**12:
        if time.time() > deadline - 10:
            break
        
        current_R = current_L + seg_size - 1
        phi_arr = compute_phi_segmented(current_L, current_R)
        phi_list = phi_arr.tolist()
        
        k, idx = find_best_run(phi_list, current_L)
        if k > best_k:
            best_k = k
            best_start = idx
        
        current_L += stride
    
    return {'start': best_start, 'length': best_k}
