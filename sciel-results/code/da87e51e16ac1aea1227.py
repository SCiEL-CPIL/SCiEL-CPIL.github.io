import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # 1. Sieve primes up to 10^6
    limit = 10**6
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.array(np.where(is_prime)[1])
    
    best_k = 1
    best_s = 1
    
    def compute_phi_segment(start, end):
        """
        Compute phi(n) for n in [start, end] using segmented sieve with NumPy.
        """
        n = end - start + 1
        phi = np.arange(start, end + 1, dtype=np.int64)
        
        # Apply small primes
        for p in primes:
            if p * p > end:
                break
            # Find first multiple of p in [start, end]
            first = (start + p - 1) // p * p
            if first > end:
                continue
            # Indices in the phi array
            # We need to update phi[first], phi[first+p], ...
            # Vectorized update
            indices = np.arange(first, end + 1, p) - start
            phi[indices] = (phi[indices] // p) * (p - 1)
            
        # Handle primes in the range
        # If phi[i] is still equal to start+i, then start+i was not divided by any p <= sqrt(end)
        # This implies start+i is prime (or 1)
        # Note: 1 is not prime, phi(1)=1. If start=1, phi[0]=1. 1 is not changed. 
        # If n > 1 and unchanged, n is prime, phi(n) = n - 1.
        
        original_vals = np.arange(start, end + 1, dtype=np.int64)
        mask = phi == original_vals
        
        if np.any(mask):
            # Get the original values
            orig_vals = original_vals[mask]
            # phi(n) = n - 1 for prime n > 1
            # phi(1) = 1
            new_phi_vals = orig_vals - 1
            new_phi_vals[orig_vals == 1] = 1
            phi[mask] = new_phi_vals
            
        return phi

    def find_max_distinct_run(phi_vals, base_start):
        """
        Find the longest run of distinct values in phi_vals.
        Returns (length, start_offset).
        """
        if len(phi_vals) == 0:
            return 0, 0
            
        seen = {}
        left = 0
        max_len = 0
        max_start = 0
        
        for right in range(len(phi_vals)):
            val = phi_vals[right]
            
            if val in seen:
                # Move left pointer until the duplicate is removed
                while phi_vals[left] != val:
                    seen[phi_vals[left]] -= 1
                    if seen[phi_vals[left]] == 0:
                        del seen[phi_vals[left]]
                    left += 1
                # Remove the duplicate at left
                del seen[val]
                left += 1
            else:
                seen[val] = 1
                
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
                max_start = left
                
        return max_len, max_start

    # Search strategy
    # Try different segment sizes and locations
    # We want to maximize k.
    
    # Segment sizes to try
    # Small segments are fast, allow more samples.
    # Large segments might contain longer runs if they exist.
    
    segment_sizes = [10**5, 2*10**5, 5*10**5, 10**6, 2*10**6]
    
    # Locations to try
    # 1. Start near 1
    # 2. Start near 10^12
    # 3. Random locations in between
    
    locations = []
    # Add small start
    for size in segment_sizes:
        locations.append((1, size))
        
    # Add near max
    max_val = 10**12
    for size in segment_sizes:
        start = max_val - size + 1
        if start >= 1:
            locations.append((start, size))
            
    # Add some random locations
    # To ensure we cover the space, we can pick random starts
    # But random might miss. Let's pick spaced locations.
    # Step = 10^10? That's 100 steps.
    # If we do 10 steps of 10^11, that's 10 segments.
    # Let's try a few specific ones.
    
    test_starts = [
        10**6,
        10**7,
        10**8,
        10**9,
        10**10,
        10**11,
        5*10**11,
        9*10**11
    ]
    
    for s in test_starts:
        for size in [10**5, 10**6]: # Use smaller sizes for many locations
             if s + size - 1 <= max_val:
                 locations.append((s, size))
    
    # Deduplicate locations? Not strictly necessary if we just check time.
    
    # Sort by potential? No, just iterate.
    # It's better to try larger sizes at specific promising spots?
    # Actually, if a run of length 200 exists, it's likely in a "dense" area of numbers.
    # Let's just iterate through the generated locations.
    
    # To avoid redundant work, we can just pick a set of unique (start, size) pairs.
    # Let's simplify: Try a fixed set of segments.
    
    segments_to_try = [
        (1, 10**6),
        (10**6, 10**6),
        (10**7, 10**6),
        (10**8, 10**6),
        (10**9, 10**6),
        (10**10, 10**6),
        (10**11, 10**6),
        (10**12 - 10**6, 10**6),
        # Try a larger segment in the middle
        (5*10**11, 2*10**6),
        (5*10**11 - 10**6, 2*10**6)
    ]
    
    # Also add some random ones if time permits
    # But first, execute the deterministic ones.
    
    for start, size in segments_to_try:
        if time.time() - start_time > deadline - 5:
            break
            
        if start < 1:
            start = 1
        if start + size - 1 > 10**12:
            size = 10**12 - start + 1
            if size <= 0:
                continue
                
        try:
            phi_vals = compute_phi_segment(start, start + size - 1)
            k, offset = find_max_distinct_run(phi_vals, start)
            
            if k > best_k:
                best_k = k
                best_s = start + offset
                
            # If we found the max possible, break
            if best_k >= 200:
                break
                
        except Exception as e:
            # Handle errors gracefully
            pass
            
    return {'start': best_s, 'length': best_k}

