import time
from math import isqrt
import numpy as np

def _is_powerful(n):
    """Check if n is a powerful number (every prime factor appears with exponent >= 2)."""
    if n < 1:
        return False
    if n == 1:
        return True
    temp = n
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        if p * p > temp:
            break
        if temp % (p * p) == 0:
            while temp % p == 0:
                temp //= p
        elif temp % p == 0:
            return False
    if temp > 1:
        r = isqrt(temp)
        if r * r != temp:
            return False
    return True

def _generate_powerful_numbers(limit):
    """Generate all powerful numbers up to limit."""
    powerful = set()
    powerful.add(1)
    
    # Powerful numbers are of the form a^2 * b^3 where b is square-free
    max_b = int(limit ** (1/3)) + 2
    
    # Pre-compute small primes for square-free check
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    for b in range(1, max_b + 1):
        t = b
        is_sf = True
        for p in small_primes:
            if p * p > t:
                break
            if t % (p * p) == 0:
                is_sf = False
                break
            while t % p == 0:
                t //= p
        if not is_sf:
            continue
        if t > 1:
            r = isqrt(t)
            if r * r == t:
                is_sf = False
        if not is_sf:
            continue
            
        b3 = b ** 3
        if b3 > limit:
            continue
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 20s safety margin before 120s limit
    
    # Generate powerful numbers
    # L = 5 * 10^8 should be fast enough for generation and counting with NumPy chunks
    pow_limit = 5 * 10**8
    powerful = _generate_powerful_numbers(pow_limit)
    K = len(powerful)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 30 or K < 10:
        # Fallback
        return {'N': 5, 'pairs': [[1, 4]]}
    
    # Convert to NumPy array
    pow_array = np.array(powerful, dtype=np.int64)
    
    # Count sums using NumPy chunks
    # We want to count a + b for all a, b in pow_array
    # We process a in chunks to limit memory usage of the sum matrix
    
    counts = {}
    chunk_size = 2000
    
    for i in range(0, K, chunk_size):
        chunk = pow_array[i : i + chunk_size]
        # Compute all sums between chunk and the full array
        # Shape: (chunk_size, K)
        sums = chunk[:, None] + pow_array[None, :]
        
        # Get unique sums and their counts in this chunk
        unique_sums, local_counts = np.unique(sums, return_counts=True)
        
        # Update global counts
        # This loop is the bottleneck. 
        # If unique_sums is large (e.g., 10 million), this takes time.
        # We can try to optimize by only updating if local_counts > 0 (always true)
        for u, c in zip(unique_sums, local_counts):
            counts[u] = counts.get(u, 0) + int(c)
            
        # Check time
        if time.time() > deadline - 20:
            break
            
    if not counts:
        return {'N': 5, 'pairs': [[1, 4]]}
        
    # Find the N with the maximum count
    # Note: The count here is for ordered pairs (a,b) including a>b.
    # We want distinct pairs a <= b.
    # As derived, maximizing the ordered count is the correct strategy.
    best_N = max(counts, key=counts.get)
    best_count_ordered = counts[best_N]
    
    # Extract pairs a <= b such that a + b = best_N
    pow_set = set(powerful)
    pairs = []
    for a in powerful:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in pow_set and a <= b:
            pairs.append([a, b])
            
    # Verify
    verified_pairs = []
    for p in pairs:
        if _is_powerful(p[0]) and _is_powerful(p[1]):
            verified_pairs.append(p)
            
    if not verified_pairs:
        verified_pairs = [[1, 4]]
        best_N = 5
        
    return {'N': best_N, 'pairs': verified_pairs}
