"""Improved program: NumPy vectorized segmented sieve for long runs of distinct totients."""
import time
import math
import random
import sys
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    # Precompute small primes for trial division and segmented sieve
    def sieve_primes(limit):
        is_prime = np.ones(limit + 1, dtype=np.uint8)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = 0
        return np.array([i for i in range(2, limit + 1) if is_prime[i]], dtype=np.int64)

    small_primes = sieve_primes(100000)  # primes up to 100000

    def phi_trial(x, primes=None):
        """Compute phi(x) using trial division by primes."""
        if primes is None:
            primes = small_primes
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result

    def find_longest_distinct_run(phi_vals, offset=0):
        """Find the longest run of consecutive distinct values in phi_vals.
        Returns (best_k, best_start) where best_start is the actual number (offset + index)."""
        n = len(phi_vals)
        best_k = 0
        best_start = offset
        # Sliding window: maintain left pointer and count of each value in window
        from collections import defaultdict
        counts = defaultdict(int)
        left = 0
        for right in range(n):
            v = phi_vals[right]
            counts[v] += 1
            # Shrink window until all values are distinct
            while counts[v] > 1:
                counts[phi_vals[left]] -= 1
                if counts[phi_vals[left]] == 0:
                    del counts[phi_vals[left]]
                left += 1
            # Window [left, right] has all distinct values
            k = right - left + 1
            if k > best_k:
                best_k = k
                best_start = offset + left

        return best_k, best_start

    def check_window(start, length, primes=None):
        """Check if a window has all distinct phi values, return max distinct prefix length."""
        if primes is None:
            primes = small_primes
        seen = set()
        k = 0
        for i in range(length):
            v = phi_trial(start + i, primes)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    def segmented_phi_numpy(low, high, primes=None):
        """Compute phi(n) for all n in [low, high] using NumPy segmented sieve.
        Returns a numpy array of phi values indexed by (n - low)."""
        if primes is None:
            primes = small_primes
        
        n = high - low + 1
        # Initialize phi array with original values
        phi = np.arange(low, high + 1, dtype=np.int64)
        
        # For each prime p <= sqrt(high), apply the (1 - 1/p) factor to multiples of p
        max_p = int(math.isqrt(high)) + 1
        for p in primes:
            if p > max_p:
                break
            p = int(p)
            # Find first multiple of p in [low, high]
            first = ((low + p - 1) // p) * p
            if first < p:
                first = p
            if first > high:
                continue
            # Apply phi[j] -= phi[j] // p for all multiples of p in range
            indices = np.arange(first - low, n, p)
            phi[indices] = phi[indices] - phi[indices] // p
        
        # Handle remaining prime factors > sqrt(high)
        # After processing all p <= sqrt(high), any remaining factor y > 1 in n
        # must be prime (since if y = a*b with a,b > sqrt(high), then y > high >= n, contradiction)
        # So we need to find, for each n, the largest prime factor > sqrt(high)
        # Efficient approach: track reduced value by dividing out small primes
        
        reduced = np.arange(low, high + 1, dtype=np.int64)
        for p in primes:
            if p > max_p:
                break
            p = int(p)
            first = ((low + p - 1) // p) * p
            if first < p:
                first = p
            if first > high:
                continue
            indices = np.arange(first - low, n, p)
            # Divide out all powers of p from reduced
            while True:
                mask = reduced[indices] % p == 0
                if not np.any(mask):
                    break
                reduced[indices[mask]] = reduced[indices[mask]] // p
        
        # For positions where reduced > 1, that's a prime factor > sqrt(high)
        mask = reduced > 1
        phi[mask] = phi[mask] - phi[mask] // reduced[mask]
        
        return phi

    best_k = 1
    best_start = 1

    # Phase 1: Segmented sieve search using NumPy over large ranges
    CHUNK_SIZE = 5 * 10**5  # Start with 500k to balance memory and speed
    
    # Search from 1 up to as far as time permits
    search_start = 1
    while time_remaining() > 15.0:
        chunk_end = min(search_start + CHUNK_SIZE - 1, 10**12)
        try:
            phi_vals = segmented_phi_numpy(search_start, chunk_end)
            k, s = find_longest_distinct_run(phi_vals, search_start)
            if k > best_k:
                best_k = k
                best_start = s
                if best_k >= 200:
                    break
        except MemoryError:
            # Try smaller chunk
            CHUNK_SIZE = 100000
            continue
        
        search_start = chunk_end + 1
        if search_start > 10**12:
            break

    # Phase 2: Targeted search in specific regions
    random.seed(42)
    
    # Try windows around interesting locations
    primorials = [2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870, 6469693230]
    
    for prim in primorials:
        if time_remaining() < 5.0:
            break
        max_mult = min(10**12 // prim, 100000)
        for _ in range(min(50, max_mult)):
            if time_remaining() < 3.0:
                break
            mult = random.randint(1, max_mult)
            start = mult * prim
            if start < 1 or start + 200 > 10**12:
                continue
            k = check_window(start, 200)
            if k > best_k:
                best_k = k
                best_start = start
            for offset in [-10, -5, -1, 1, 5, 10]:
                s2 = start + offset
                if s2 < 1 or s2 + 200 > 10**12:
                    continue
                k2 = check_window(s2, 200)
                if k2 > best_k:
                    best_k = k2
                    best_start = s2

    # Phase 3: Random search in mid-range
    if time_remaining() > 5.0 and best_k < 200:
        for _ in range(200):
            if time_remaining() < 2.0:
                break
            s = random.randint(10**6, 10**11)
            if s + 200 > 10**12:
                s = 10**12 - 200
            k = check_window(s, 200)
            if k > best_k:
                best_k = k
                best_start = s

    # Phase 4: Local refinement around best found
    if best_k > 1 and time_remaining() > 3.0:
        for delta in range(-50, 51):
            if time_remaining() < 1.0:
                break
            s = best_start + delta
            if s < 1 or s + 200 > 10**12:
                continue
            k = check_window(s, 200)
            if k > best_k:
                best_k = k
                best_start = s

    # Ensure constraints
    if best_k > 200:
        best_k = 200
    if best_start + best_k - 1 > 10**12:
        best_start = 10**12 - best_k + 1
    if best_start < 1:
        best_start = 1
    if best_k < 1:
        best_k = 1

    return {'start': best_start, 'length': best_k}
# EVOLVE-BLOCK-END
