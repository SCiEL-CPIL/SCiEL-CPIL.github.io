import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # 1. Generate primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    def compute_phi_block(start, length):
        """
        Computes phi(n) for n in [start, start+length-1] using segmented sieve.
        Returns a list of length 'length' containing phi values.
        """
        phi_vals = [0] * length
        rem_vals = [0] * length
        for i in range(length):
            n = start + i
            phi_vals[i] = n
            rem_vals[i] = n
            
        # Sieve with primes
        # We only need primes up to sqrt(start + length - 1)
        # Since start is large, sqrt is ~10^6.
        # But for small length, we can just iterate through all primes up to 10^6
        # and check if they divide any number in the range.
        # This is O(pi(10^6)) which is ~78k iterations.
        
        end = start + length - 1
        sqrt_end = int(math.isqrt(end))
        
        for p in primes:
            if p > sqrt_end:
                break
            
            # Find the first multiple of p in [start, end]
            # first_multiple = ceil(start / p) * p
            first_multiple = ((start + p - 1) // p) * p
            
            if first_multiple > end:
                continue
                
            # Iterate through multiples of p in the range
            # Since length is small (e.g., 300), there are few multiples.
            for j in range(first_multiple - start, length, p):
                if phi_vals[j] % p == 0:
                    phi_vals[j] -= phi_vals[j] // p
                while rem_vals[j] % p == 0:
                    rem_vals[j] //= p
        
        # Handle remaining prime factors (which must be > sqrt_end)
        for i in range(length):
            if rem_vals[i] > 1:
                phi_vals[i] -= phi_vals[i] // rem_vals[i]
                
        return phi_vals

    def find_best_run_in_block(block_start, block_len):
        """
        Finds the longest run of distinct phi values in the block.
        Returns (best_k, best_start_offset).
        """
        if time.time() > deadline:
            return 0, 0
            
        phis = compute_phi_block(block_start, block_len)
        
        max_k = 0
        max_start = 0
        seen = set()
        cur_k = 0
        cur_start = 0
        
        for i in range(block_len):
            if phis[i] not in seen:
                if cur_k == 0:
                    cur_start = i
                cur_k += 1
                seen.add(phis[i])
                if cur_k > max_k:
                    max_k = cur_k
                    max_start = cur_start
            else:
                # Collision, reset
                seen = set()
                cur_k = 0
                
        # Cap at 200
        if max_k > 200:
            max_k = 200
            # We need to ensure the run starting at max_start has length 200
            # If the run was longer, the first 200 are distinct.
            # So returning max_start with length 200 is valid.
            
        return max_k, max_start

    best_k = 1
    best_s = 1
    
    # Search strategy:
    # High ranges have low collision probability.
    # Try a few high start points.
    # Block size 300 is sufficient to find a run of 200 if one exists nearby.
    
    candidates = [
        10**12 - 1000,
        10**12 - 2000,
        10**12 - 3000,
        10**11,
        5 * 10**11
    ]
    
    for s in candidates:
        if time.time() > deadline:
            break
        
        # Ensure s + 299 <= 10^12
        if s + 299 > 10**12:
            s = 10**12 - 299
            
        block_len = 300
        k, offset_in_block = find_best_run_in_block(s, block_len)
        
        if k > best_k:
            best_k = k
            best_s = s + offset_in_block
            
            # If we found a run of 200, we are done (max possible)
            if best_k >= 200:
                return {'start': best_s, 'length': best_k}

    # If we didn't find 200, try a slightly larger search if time permits
    # But the above should suffice given the low collision rate.
    
    # Safety check: ensure best_k is within [1, 200]
    if best_k > 200:
        best_k = 200
    if best_k < 1:
        best_k = 1
        best_s = 1 # Fallback to valid minimal

    return {'start': best_s, 'length': best_k}
