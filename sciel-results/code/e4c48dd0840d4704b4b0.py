import math
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # Safety margin
    
    def time_exceeded():
        return time.time() >= DEADLINE

    def generate_powerful(limit):
        """Generate all powerful numbers up to limit."""
        if limit < 1:
            return []
        
        powerful = [1]
        powerful_set = {1}
        
        # Get primes up to sqrt(limit)
        # Actually, we need primes up to limit for the base, but exponents >= 2
        # So p^2 <= limit => p <= sqrt(limit)
        sqrt_limit = int(math.isqrt(limit))
        if sqrt_limit < 2:
            return [1]
        
        # Sieve of Eratosthenes
        sieve = bytearray(b'\x01') * (sqrt_limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(math.isqrt(sqrt_limit)) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        primes = [i for i in range(2, sqrt_limit + 1) if sieve[i]]
        
        # Recursive generation
        def gen(idx, val):
            if val > limit:
                return
            if val > 1:
                powerful_set.add(val)
            
            for i in range(idx, len(primes)):
                p = primes[i]
                nv = val * p * p
                if nv > limit:
                    break
                gen(i + 1, nv)
                while nv * p <= limit:
                    nv *= p
                    powerful_set.add(nv)
                    # We don't need to recurse further with the same p
                    # because the next prime must be larger
                    # But we do need to add nv to the set
                    # And we can't use p again for the next factor
                    # The recursion handles the "next prime" part
                    # Wait, the loop `for i in range(idx, ...)` handles the next prime.
                    # The while loop just adds higher powers of p.
                    # But we must not multiply by p again in the next recursive step
                    # The recursive step starts at i+1.
                    # So this is correct.
        
        gen(0, 1)
        return list(powerful_set)

    # Generate powerful numbers
    # We want about 20,000 - 30,000 numbers
    # P(L) ~ 1.2 L^(2/3)
    # 25,000 ~ 1.2 L^(2/3) => L^(2/3) ~ 20,000 => L ~ 20,000^1.5 ~ 2.8 * 10^6
    # Let's use L = 10^7 to be safe and have plenty.
    L = 10**7
    pn_list = generate_powerful(L)
    pn_list.sort()
    
    if time_exceeded():
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Determine K
    # Max K such that K^2 * 4 bytes < 2 GB => K^2 < 5 * 10^8 => K < 22000
    # Let's use K = 20000
    K = min(len(pn_list), 20000)
    small_pn = np.array(pn_list[:K], dtype=np.int32)
    
    if time_exceeded():
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Compute sums
    # Use broadcasting
    sums = small_pn[:, None] + small_pn[None, :]
    
    if time_exceeded():
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Count occurrences
    # sums is int32
    max_sum = int(sums.max())
    counts = np.bincount(sums.flat, minlength=max_sum + 1)
    
    if time_exceeded():
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Find best N
    # We want to maximize R(N) = (C(N) + D(N)) / 2
    # C(N) is counts[N]
    # D(N) is 1 if N is even and N/2 is in small_pn, else 0
    # Since D(N) is small, we can just look at top C(N)
    
    # Get top indices
    # np.argsort is slow for large arrays? 
    # counts size is ~ 2 * 10^7.
    # argsort takes ~ 1s?
    
    # Alternative: just find the max
    # But there might be ties.
    # Let's get the top 10 candidates.
    
    # To avoid full sort, we can use np.argpartition
    # Or just iterate?
    
    # Let's try to find the max directly
    # But we need to check D(N) adjustment.
    
    # Let's get the top 100 sums by count
    # kth = len(counts) - 100
    # top_indices = np.argpartition(counts, kth)[-100:]
    
    # Actually, just finding the max is O(M).
    # But we need the pairs.
    
    best_N = -1
    best_R = -1
    best_pairs = []
    
    # We can just check the top few candidates
    # Let's find the index of the max count
    # If there are multiple, we might miss a better one with D=1 vs D=0
    # But D is at most 1.
    # So if C(N1) > C(N2) + 2, N1 is definitely better.
    # We can just take the top 5 by C(N) and evaluate R(N) for them.
    
    # Get top 5 indices
    # Using argpartition is efficient
    k = 5
    if len(counts) > k:
        # We want the largest k values
        # argpartition returns indices that would sort the array
        # We want the last k indices of the partition
        part_indices = np.argpartition(counts, -k)[-k:]
        # Sort these k indices by count descending
        part_indices = part_indices[counts[part_indices][::-1].argsort()]
    else:
        part_indices = np.argsort(counts)[::-1][:k]
        
    for N in part_indices:
        N = int(N)
        C = int(counts[N])
        D = 0
        if N % 2 == 0:
            half = N // 2
            # Check if half is in small_pn
            # small_pn is sorted
            # Use binary search
            idx = np.searchsorted(small_pn, half)
            if idx < len(small_pn) and small_pn[idx] == half:
                D = 1
        R = (C + D) // 2
        if R > best_R:
            best_R = R
            best_N = N
            # Reconstruct pairs
            pairs = []
            small_pn_set = set(small_pn.tolist())
            for a in small_pn:
                b = N - a
                if b < a:
                    break
                if b in small_pn_set:
                    pairs.append([int(a), int(b)])
            best_pairs = pairs
    
    if best_N == -1:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_N, 'pairs': best_pairs}

# EVOLVE-BLOCK-END
