import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 118.0  # safety margin before 120s kill

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # Step 1: Sieve primes up to 10^6 for fast factorization
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Convert primes to numpy array for vectorized operations
    primes_np = np.array(primes, dtype=np.int64)

    def compute_phi_segment(start, length):
        """
        Compute phi for all numbers in [start, start+length) using segmented sieve with NumPy.
        Returns list of phi values.
        """
        if length <= 0:
            return []
        
        # Initialize with the number itself
        # Use int64 to avoid overflow in intermediate steps if possible, though phi(n) < n <= 10^12 fits in int64
        phi = np.arange(start, start + length, dtype=np.int64)
        remaining = phi.copy()
        
        # For each prime, divide out its contribution
        # We only need primes up to sqrt(max_n) ~ 10^6
        for p in primes_np:
            if p >= start + length:
                break
            
            # Find first multiple of p in [start, start+length)
            first = ((start + p - 1) // p) * p
            if first < start:
                first += p
            
            if first >= start + length:
                continue

            # Indices of multiples of p in the segment
            indices = np.arange(first - start, length, p, dtype=np.int64)
            
            # Check which of these still have p as a factor in remaining
            # We need to update phi only if p divides remaining[idx]
            # Since we process primes in increasing order, remaining[idx] might have had smaller primes divided out.
            # If p divides remaining[idx], then p is a prime factor of the original number.
            
            rem_vals = remaining[indices]
            mask = (rem_vals % p == 0)
            
            if not np.any(mask):
                continue
                
            # Update phi: phi -= phi // p
            idxs = indices[mask]
            phi[idxs] -= phi[idxs] // p
            
            # Update remaining: divide by p as many times as possible
            # This is tricky to vectorize perfectly for "while" loop, but we can do:
            # rem_vals = remaining[indices]
            # For those divisible by p, divide by p.
            # We can just do remaining[indices] //= p for those that are divisible?
            # No, we need to remove ALL factors of p.
            # Since we only care if p is a factor for the phi update, and we only process each prime once,
            # we just need to ensure that for the NEXT prime q, p is not a factor.
            # So we must remove all factors of p from remaining.
            
            # Vectorized division by p while divisible is hard.
            # But note: we only need to know if p divides remaining for the *current* prime p.
            # For subsequent primes q, we need remaining to not be divisible by p.
            # So we must divide remaining by p completely.
            
            # Optimization: Since p is prime, we can just do:
            # vals = remaining[idxs]
            # while np.any(vals % p == 0): vals[vals % p == 0] //= p
            # This loop might be slow if there are many powers.
            # However, for random numbers, the power is usually 1 or 2.
            # Let's try a simple loop for the division part, but only on the subset.
            
            vals = remaining[idxs]
            # Divide by p until not divisible
            # This is a bit expensive if done in Python loop for each index, but we are doing it for a subset.
            # Can we vectorize the "while"?
            # Not easily.
            # Alternative: Just divide by p once? No, must remove all.
            # Let's stick to the loop for the division, but it's on a small subset usually.
            # Actually, we can just do:
            # remaining[idxs] = np.array([v // (p ** (v.bit_length() - (v//p).bit_length())) for v in vals]) # No, too complex.
            
            # Simple approach:
            # The number of times p divides v is small.
            # Let's just do a few iterations of vectorized division.
            # Most numbers have p^1. Some p^2. Very few p^3.
            for _ in range(10): # Max power of 2 for 10^12 is ~40, but for large p it's 1.
                mask2 = (vals % p == 0)
                if not np.any(mask2):
                    break
                vals[mask2] //= p
            remaining[idxs] = vals

        # Handle remaining factors > sqrt(10^12)
        # These are primes (or 1)
        rem_final = remaining
        mask_large = rem_final > 1
        if np.any(mask_large):
            idxs_large = np.where(mask_large)[0]
            phi[idxs_large] -= phi[idxs_large] // rem_final[idxs_large]
        
        return phi.tolist()

    def find_best_run_sliding(start, length, max_k=200):
        """
        Find the longest run of distinct phi values in [start, start+length)
        using a sliding window approach for O(n) instead of O(n^2).
        """
        if length <= 0:
            return 0, start
        
        # Compute phi for the range using segmented approach
        phis = compute_phi_segment(start, length)
        
        # Sliding window: maintain a window [left, right] where all phi values are distinct
        # When we find a duplicate, shrink from the left
        best_k = 1
        best_s = start
        left = 0
        freq = {}
        
        for right in range(length):
            val = phis[right]
            freq[val] = freq.get(val, 0) + 1
            
            # Shrink window from left until all values are distinct
            while freq[val] > 1:
                left_val = phis[left]
                freq[left_val] -= 1
                if freq[left_val] == 0:
                    del freq[left_val]
                left += 1
            
            current_k = right - left + 1
            if current_k > best_k:
                best_k = current_k
                best_s = start + left
            # Cap at max_k
            if best_k >= max_k:
                break
        
        return min(best_k, max_k), best_s

    # Strategy:
    # 1. Search in large ranges near 10^12 where longer runs are more likely
    # 2. Use sliding window with segmented phi computation for efficiency
    # 3. Structured search: try specific regions known to have properties favoring distinct totients
    
    best_k = 1
    best_s = 1
    
    # First, check a small range to get a baseline
    if time_left() > 10:
        k, s = find_best_run_sliding(1, 5000, 200)
        if k > best_k:
            best_k, best_s = k, s

    # Structured search in promising regions
    # Longer runs of distinct totients are more likely in regions with:
    # - Numbers with diverse prime factorizations
    # - Avoiding arithmetic progressions with common factors
    
    random.seed(42)
    
    # Try multiple strategies:
    # 1. Random search in large ranges
    # 2. Search near powers of 2 (numbers with varied factorizations)
    # 3. Search in gaps between highly composite numbers
    
    while time_left() > 15:
        # Pick a random start in [10^9, 10^12]
        strategy = random.random()
        
        if strategy < 0.6:
            # Random search
            start = random.randint(10**9, 10**12 - 200)
        elif strategy < 0.8:
            # Near powers of 2
            exp = random.randint(30, 40)
            start = (1 << exp) + random.randint(-10000, 10000)
            start = max(10**9, min(10**12 - 200, start))
        else:
            # Near prime multiples (numbers just after primes have unique properties)
            base = random.randint(10**10, 10**12)
            start = base + random.randint(1, 1000)
            start = max(10**9, min(10**12 - 200, start))
        
        length = 2000  # Larger window to find longer runs
        k, s = find_best_run_sliding(start, length, 200)
        if k > best_k:
            best_k, best_s = k, s
    
    # Also try to extend the best run found
    if best_k < 200 and time_left() > 10:
        # Try extending left
        left_start = max(1, best_s - 2000)
        left_len = best_s - left_start + best_k + 2000
        k_left, s_left = find_best_run_sliding(left_start, left_len, 200)
        if k_left > best_k:
            best_k, best_s = k_left, s_left
        
        # Try extending right
        right_end = min(10**12, best_s + best_k + 2000)
        right_start = best_s
        right_len = right_end - right_start + 1
        k_right, s_right = find_best_run_sliding(right_start, right_len, 200)
        if k_right > best_k:
            best_k, best_s = k_right, s_right

    return {'start': best_s, 'length': best_k}
