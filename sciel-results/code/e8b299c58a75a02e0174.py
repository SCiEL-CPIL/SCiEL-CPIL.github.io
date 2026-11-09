import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Strategy: Use segmented sieve to compute phi for large ranges,
    # then find the longest run of distinct phi values.
    
    # Step 1: Get primes up to sqrt(10^12) = 10^6 using sieve of Eratosthenes
    limit = 10**6
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    primes = np.where(sieve)[0]
    
    # Step 2: Segmented computation of phi for a large range
    # We'll scan from small numbers upward, looking for long runs.
    # Key insight: phi(n) is even for n > 2, so collisions are common.
    # But we need pairwise distinct values in the run.
    
    # Let's first do a quick scan of small numbers to understand the landscape
    # Then use larger segments for more thorough search
    
    def compute_phi_segment(lo, hi, primes_arr):
        """Compute phi(n) for n in [lo, hi] using segmented sieve."""
        n = hi - lo + 1
        # phi[n-i] will store phi(lo+i)
        phi_vals = np.arange(lo, hi + 1, dtype=np.int64)
        # remaining[n-i] tracks the remaining factor
        remaining = np.arange(lo, hi + 1, dtype=np.int64)
        
        for p in primes_arr:
            if p * p > hi:
                break
            # Find first multiple of p in [lo, hi]
            first = ((lo + p - 1) // p) * p
            if first < lo:
                first += p
            # Process all multiples of p
            for j in range(first, hi + 1, p):
                idx = j - lo
                if remaining[idx] % p == 0:
                    phi_vals[idx] = phi_vals[idx] // p * (p - 1)
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        # Handle remaining prime factors
        for i in range(n):
            if remaining[i] > 1:
                phi_vals[i] = phi_vals[i] // remaining[i] * (remaining[i] - 1)
        
        return phi_vals
    
    # First, do a thorough scan of small numbers [1, 200000]
    # to find good candidate regions
    small_hi = 200000
    phi_small = compute_phi_segment(1, small_hi, primes)
    
    # Find longest run of distinct phi values in small range
    best_k = 1
    best_s = 1
    seen = set()
    k = 0
    for i in range(small_hi):
        val = phi_small[i]
        if val not in seen:
            seen.add(val)
            k += 1
            if k > best_k:
                best_k = k
                best_s = i + 1  # 1-indexed
        else:
            # Collision - reset window
            # We need to find the new start after the collision
            # Simple approach: rebuild seen from the collision point
            # But that's O(k) each time. For now, just reset.
            # Better: use a sliding window with a dict tracking counts
            pass
    
    # Let's do a proper sliding window for the small range
    best_k = 1
    best_s = 1
    count = defaultdict(int)
    left = 0
    for right in range(small_hi):
        val = phi_small[right]
        count[val] += 1
        while count[val] > 1:
            count[phi_small[left]] -= 1
            left += 1
        current_k = right - left + 1
        if current_k > best_k:
            best_k = current_k
            best_s = left + 1  # 1-indexed
    
    # Now search in larger ranges to find longer runs
    # Use larger segments
    seg_size = 500000
    search_start = small_hi + 1
    max_search = 10**7  # Search up to 10 million initially
    
    while search_start < max_search and time.time() < deadline:
        seg_end = min(search_start + seg_size - 1, max_search)
        phi_seg = compute_phi_segment(search_start, seg_end, primes)
        
        # Sliding window for distinct values
        count = defaultdict(int)
        left = 0
        for right in range(len(phi_seg)):
            val = phi_seg[right]
            count[val] += 1
            while count[val] > 1:
                count[phi_seg[left]] -= 1
                left += 1
            current_k = right - left + 1
            if current_k > best_k:
                best_k = current_k
                best_s = search_start + left
                if best_k >= 200:
                    break
        
        search_start = seg_end + 1
        if best_k >= 200:
            break
    
    # If we haven't found a long enough run, try even larger ranges
    # but be mindful of time
    if best_k < 100 and time.time() < deadline:
        # Try searching in larger gaps
        seg_size = 200000
        for base in range(10**7, min(10**9, int(deadline - time.time()) * 10000), seg_size * 10):
            if time.time() > deadline - 10:
                break
            seg_start = base
            seg_end = base + seg_size - 1
            if seg_end > 10**12:
                break
            phi_seg = compute_phi_segment(seg_start, seg_end, primes)
            
            count = defaultdict(int)
            left = 0
            for right in range(len(phi_seg)):
                val = phi_seg[right]
                count[val] += 1
                while count[val] > 1:
                    count[phi_seg[left]] -= 1
                    left += 1
                current_k = right - left + 1
                if current_k > best_k:
                    best_k = current_k
                    best_s = seg_start + left
                    if best_k >= 200:
                        break
            
            if best_k >= 200:
                break
    
    # Ensure we return a valid result
    if best_k > 200:
        best_k = 200
    
    return {'start': best_s, 'length': best_k}
