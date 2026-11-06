import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time)
    
    def compute_phi_sieve_numpy(n):
        """Compute phi(1..n) using a sieve-based approach with NumPy."""
        phi = np.arange(n + 1, dtype=np.int64)
        for i in range(2, n + 1):
            if phi[i] == i:  # i is prime
                for j in range(i, n + 1, i):
                    phi[j] = phi[j] // i * (i - 1)
        return phi
    
    def find_longest_distinct_run(phi_vals, offset=0):
        """Find the longest run of consecutive integers with distinct totients.
        phi_vals is an array where phi_vals[i] = phi(offset + i + 1) (1-indexed).
        Returns (length, start_index_in_original_numbers)"""
        n = len(phi_vals)
        if n == 0:
            return (0, 0)
        
        best_len = 1
        best_start = 0  # 0-indexed in phi_vals
        
        # For each starting position, find how far we can go with distinct values
        # We use a sliding window approach with a hash set
        # But for efficiency, we'll scan and track the longest valid window
        
        # Alternative: for each position, extend as far as possible
        # This is O(n^2) worst case but in practice much less
        
        # Better approach: use the fact that if phi[i] == phi[j] for i < j,
        # then any window containing both i and j is invalid.
        # We can use a two-pointer / sliding window technique.
        
        left = 0
        freq = {}
        max_len = 0
        max_start = 0
        
        for right in range(n):
            val = int(phi_vals[right])
            freq[val] = freq.get(val, 0) + 1
            
            while freq[val] > 1:
                left_val = int(phi_vals[left])
                freq[left_val] -= 1
                left += 1
            
            curr_len = right - left + 1
            if curr_len > max_len:
                max_len = curr_len
                max_start = left
        
        # Convert back to original number indices
        # phi_vals[0] corresponds to number offset+1
        # max_start is 0-indexed in phi_vals, so the number is offset + max_start + 1
        return (max_len, offset + max_start + 1)
    
    # Strategy: Try multiple ranges with increasing size
    # Start with a moderate range, then try larger ones if time permits
    
    best_length = 1
    best_start = 1
    
    # First, try a smaller range to get a baseline
    # Then try larger ranges
    
    # Range sizes to try (we'll stop when time runs out)
    # The key insight: longer runs are more likely in larger ranges
    # But we need to balance computation time for phi sieve vs. search time
    
    # Let's try several range sizes
    # For each, compute phi via sieve, then find longest distinct run
    
    # Start with a range that's computable quickly
    # phi sieve for n takes O(n log log n) time
    
    # Try ranges: 10^5, 10^6, 10^7, 10^8, ...
    # But we need to be careful about memory and time
    
    # Let's start with a reasonable range
    # and expand if we have time
    
    # Actually, let's think about this differently.
    # The task says s + k - 1 <= 10^12, so s can be up to 10^12.
    # But computing phi for a single number up to 10^12 is feasible with trial division
    # or Pollard's rho, but doing it for 200 consecutive numbers is slow.
    
    # The sieve approach works well for contiguous ranges starting from 1.
    # For arbitrary ranges [s, s+k-1], we'd need a different approach.
    
    # Let's first try ranges starting from 1 with increasing size.
    # Then, if we have time, try random ranges in the middle of [1, 10^12].
    
    # Phase 1: Sieve from 1 with increasing range sizes
    # Phase 2: If time permits, try targeted searches in larger ranges
    
    # Phase 1: Sieve from 1
    # Try increasing range sizes
    range_sizes = [100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000]
    
    for n in range_sizes:
        if time_remaining() < 5:
            break
        
        # Check if we can afford this sieve
        # Rough estimate: sieve for n takes about n * log(log(n)) / 10^6 seconds?
        # Let's just try and check time after
        
        try:
            phi = compute_phi_sieve_numpy(n)
        except MemoryError:
            break
        except Exception:
            break
        
        if time_remaining() < 2:
            break
        
        length, start = find_longest_distinct_run(phi, offset=0)
        
        if length > best_length:
            best_length = length
            best_start = start
        
        # If we found a run of 200, we can't do better
        if best_length >= 200:
            break
    
    # Phase 2: Try some random ranges in larger territory
    # For this, we need a fast phi computation for individual numbers
    # or a segmented sieve
    
    if time_remaining() > 10 and best_length < 200:
        # Segmented sieve for phi in a range [lo, hi]
        def segmented_phi_sieve(lo, hi):
            """Compute phi(n) for all n in [lo, hi] using segmented sieve."""
            n = hi - lo + 1
            phi = np.arange(lo, hi + 1, dtype=np.int64).copy()
            
            # We need to factor each number in [lo, hi]
            # Use primes up to sqrt(hi)
            max_p = int(np.sqrt(hi)) + 1
            
            # Generate primes up to max_p using simple sieve
            is_prime = np.ones(max_p + 1, dtype=bool)
            is_prime[0] = is_prime[1] = False
            for i in range(2, int(np.sqrt(max_p)) + 1):
                if is_prime[i]:
                    is_prime[i*i::i] = False
            primes = np.where(is_prime)[0]
            
            # For each prime, update phi values in [lo, hi]
            for p in primes:
                # Find the first multiple of p in [lo, hi]
                start = ((lo + p - 1) // p) * p
                if start < lo:
                    start = lo
                # Actually, we want multiples of p
                start = ((lo + p - 1) // p) * p
                
                for j in range(start, hi + 1, p):
                    idx = j - lo
                    if phi[idx] % p == 0:
                        phi[idx] = phi[idx] // p * (p - 1)
            
            return phi
        
        # Try a few random ranges
        import random
        random.seed(42)
        
        # Try ranges of length ~10000 in various locations
        for _ in range(20):
            if time_remaining() < 5:
                break
            
            # Choose a random starting point
            # Avoid very small numbers where runs might be shorter
            lo = random.randint(1000000, 10**12 - 10000)
            hi = lo + 9999
            
            try:
                phi = segmented_phi_sieve(lo, hi)
            except Exception:
                continue
            
            if time_remaining() < 2:
                break
            
            length, start = find_longest_distinct_run(phi, offset=lo - 1)
            
            if length > best_length:
                best_length = length
                best_start = start
            
            if best_length >= 200:
                break
    
    # Final safety check
    if best_length > 200:
        best_length = 200
        # Adjust start if needed
        if best_start + best_length - 1 > 10**12:
            best_start = 10**12 - best_length + 1
        if best_start < 1:
            best_start = 1
    
    return {'start': best_start, 'length': best_length}
