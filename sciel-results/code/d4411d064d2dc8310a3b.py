import time
import sys
import numpy as np
from math import isqrt

def _is_powerful(n):
    """Check if n is a powerful number (every prime factor appears with exponent >= 2)."""
    if n < 1:
        return False
    if n == 1:
        return True
    temp = n
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)
    for p in small_primes:
        pp = p * p
        if pp > temp:
            break
        if temp % pp == 0:
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
    """Generate all powerful numbers up to limit using n = a^2 * b^3, b square-free."""
    powerful = set()
    powerful.add(1)
    max_b = int(limit ** (1.0 / 3.0)) + 2

    # Sieve of Eratosthenes up to max_b to get primes
    max_b_plus = max_b + 1
    if max_b_plus > 1:
        is_prime = bytearray(b'\x01') * (max_b_plus)
        is_prime[0] = 0
        if max_b_plus > 1:
            is_prime[1] = 0
        for i in range(2, isqrt(max_b) + 1):
            if is_prime[i]:
                for j in range(i * i, max_b_plus, i):
                    is_prime[j] = 0
        small_primes = [i for i in range(2, max_b_plus) if is_prime[i]]
    else:
        small_primes = []

    for b in range(1, max_b + 1):
        # Check if b is square-free
        t = b
        is_sf = True
        for p in small_primes:
            if p * p > t:
                break
            if t % (p * p) == 0:
                is_sf = False
                break
        if not is_sf:
            continue

        b3 = b * b * b
        if b3 > limit:
            continue
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
            else:
                break
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit

    # Strategy: Generate powerful numbers up to a limit, then use a smarter
    # approach to find N with many representations as sum of two powerful numbers.
    #
    # Key insight: Instead of computing ALL pair sums (O(K^2)), we can:
    # 1. Generate powerful numbers up to L
    # 2. For candidate N values, count how many pairs (a, b) with a+b=N exist
    # 3. Use the fact that if N is around 2L, then a and b are both around L,
    #    and we can use two-pointer or binary search for each candidate
    #
    # Better approach: Use the "additive energy" concept.
    # For each powerful number a, the number of b such that a+b=N is 1 if N-a is powerful.
    # So for a fixed N, the count is the number of a in P such that N-a is also in P.
    #
    # To find good N values, we can:
    # 1. Look at the density of powerful numbers
    # 2. Try N values where the local density is high
    #
    # Even better: Use FFT-based convolution if the range is manageable.
    # The sum a+b = N means we're looking at the convolution of the indicator
    # function of P with itself, evaluated at N.
    #
    # For L = 400M, the range [0, 2L] has 800M+1 points, which is too large for FFT.
    # But we can use a smaller effective range by noting that most powerful numbers
    # are clustered, or we can use a different approach.
    #
    # Let's try: Generate powerful numbers up to L=200M (K~30000).
    # Then use chunked outer sum with numpy, but be smarter about which N to check.
    #
    # Actually, the current approach with bincount is already O(K^2) but vectorized.
    # The issue is memory. Let's try to optimize the memory usage.
    #
    # Alternative: For each chunk of a's, compute sums with all b's, but use
    # a smaller max_sum by only considering N in a reasonable range.
    #
    # Let's try L = 300M and see if we can fit in memory with careful chunking.

    pow_limit = 300_000_000  # 3 * 10^8

    powerful = _generate_powerful_numbers(pow_limit)
    K = len(powerful)

    elapsed = time.time() - start_time
    if time.time() > deadline - 80:
        # Fallback to smaller limit if generation was unexpectedly slow
        pow_limit = 100_000_000
        powerful = _generate_powerful_numbers(pow_limit)
        K = len(powerful)
        elapsed = time.time() - start_time

    # Convert to numpy array for fast operations
    pow_arr = np.array(powerful, dtype=np.int64)

    # We need to count sums a+b for all a,b in pow_arr.
    # Max sum is 2 * pow_limit.
    max_sum = 2 * pow_limit + 1

    # Use int32 for counts
    counts = np.zeros(max_sum, dtype=np.int32)

    # Chunked processing to manage memory
    # Memory for one chunk of size C: K * C * 8 bytes
    # If K=30000, C=1000, Mem = 30000 * 1000 * 8 = 240 MB. Safe.
    chunk_size = 1000

    for i in range(0, K, chunk_size):
        chunk = pow_arr[i:i + chunk_size]
        # Compute outer sum: shape (K, chunk_size)
        sums = pow_arr[:, None] + chunk[None, :]

        # Flatten and bincount
        chunk_counts = np.bincount(sums.ravel(), minlength=max_sum)
        counts += chunk_counts

        # Time check
        if time.time() > deadline - 20:
            break

    if not any(counts > 0):
        return {'N': 5, 'pairs': [[1, 4]]}

    # Find the best N
    best_N = int(np.argmax(counts))
    best_count = int(counts[best_N])

    # Extract pairs
    pow_set = set(powerful)
    pairs = []

    # We only need a <= b, so a <= best_N / 2
    half_N = best_N // 2
    for a in powerful:
        if a > half_N:
            break
        b = best_N - a
        if b in pow_set:
            if _is_powerful(a) and _is_powerful(b):
                pairs.append([a, b])

    if not pairs:
        # Fallback
        return {'N': 5, 'pairs': [[1, 4]]}

    return {'N': best_N, 'pairs': pairs}
