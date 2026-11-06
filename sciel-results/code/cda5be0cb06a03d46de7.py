import time
import math

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s
    
    def get_primes_up_to(limit):
        """Sieve of Eratosthenes to get primes up to limit."""
        if limit < 2:
            return []
        is_prime = bytearray([1]) * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    def segmented_phi(start, end, primes):
        """Compute phi for all numbers in [start, end] using segmented sieve."""
        n = end - start + 1
        phi = list(range(start, end + 1))
        is_prime_mark = bytearray([1]) * n  # track if still "prime" in segment
        
        for p in primes:
            if p * p > end:
                break
            # Find first multiple of p in [start, end]
            first = ((start + p - 1) // p) * p
            if first < p * p:
                first = p * p  # don't mark p itself if p is in range
            if first > end:
                continue
            for j in range(first, end + 1, p):
                idx = j - start
                phi[idx] -= phi[idx] // p
                is_prime_mark[idx] = 0
        
        # For remaining numbers that are still marked as prime (i.e., > sqrt(end) and prime)
        for i in range(n):
            if is_prime_mark[i] and phi[i] == start + i:
                # This is a prime
                phi[i] = start + i - 1
            elif phi[i] == start + i:
                # This shouldn't happen for composites, but handle edge case
                pass
        
        # Actually, the segmented approach above has issues. Let me use a cleaner method.
        return phi
    
    def compute_phi_segment(start, end, primes):
        """Cleaner segmented phi computation."""
        n = end - start + 1
        phi = list(range(start, end + 1))
        
        for p in primes:
            if p > end:
                break
            # Find first multiple of p in [start, end]
            if p >= start:
                first = p
            else:
                first = ((start + p - 1) // p) * p
            
            if first > end:
                continue
                
            for j in range(first, end + 1, p):
                idx = j - start
                if phi[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
        
        return phi
    
    def longest_distinct_run(values):
        """Find longest contiguous subarray with all distinct values using sliding window."""
        n = len(values)
        if n == 0:
            return 0, 0
        
        seen = {}
        left = 0
        best_len = 0
        best_start = 0
        
        for right in range(n):
            val = values[right]
            if val in seen and seen[val] >= left:
                left = seen[val] + 1
            seen[val] = right
            
            curr_len = right - left + 1
            if curr_len > best_len:
                best_len = curr_len
                best_start = left
        
        return best_len, best_start
    
    # Get primes up to sqrt(10^12) = 10^6
    prime_limit = 1000000
    primes = get_primes_up_to(prime_limit)
    
    if time.time() > deadline:
        return {'start': 1, 'length': 1}
    
    best_k = 1
    best_s = 1
    
    # Search strategy: try different ranges
    # Start with smaller ranges to get quick results, then expand
    
    # Phase 1: Search first 100,000
    segment_size = 100000
    for seg_start in [1]:
        seg_end = min(seg_start + segment_size - 1, 10**12)
        phi_vals = compute_phi_segment(seg_start, seg_end, primes)
        k, s = longest_distinct_run(phi_vals)
        if k > best_k:
            best_k = k
            best_s = seg_start + s
        
        if time.time() > deadline:
            break
    
    # Phase 2: Try larger segments at different positions
    # The longest runs might be at larger values
    search_positions = [
        10**6, 10**7, 10**8, 10**9, 10**10, 10**11
    ]
    
    for pos in search_positions:
        if time.time() > deadline - 5:
            break
        
        seg_end = min(pos + segment_size - 1, 10**12)
        if pos > 10**12:
            continue
            
        phi_vals = compute_phi_segment(pos, seg_end, primes)
        k, s = longest_distinct_run(phi_vals)
        if k > best_k:
            best_k = k
            best_s = pos + s
    
    # Phase 3: If we have time, try even larger segments
    if time.time() < deadline - 10:
        # Try a very large segment
        large_size = 1000000
        for pos in [10**5, 10**6, 10**7, 10**8]:
            if time.time() > deadline - 5:
                break
            seg_end = min(pos + large_size - 1, 10**12)
            if pos > 10**12:
                continue
            phi_vals = compute_phi_segment(pos, seg_end, primes)
            k, s = longest_distinct_run(phi_vals)
            if k > best_k:
                best_k = k
                best_s = pos + s
    
    # Ensure k <= 200
    best_k = min(best_k, 200)
    best_s = max(1, best_s)
    
    # Final safety check
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1
        best_s = 1
    
    return {'start': best_s, 'length': best_k}
