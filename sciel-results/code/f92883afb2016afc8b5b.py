import time
from math import isqrt
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Use full budget with safety margin
    
    def time_left():
        return deadline - time.time()
    
    # Step 1: Generate all powerful numbers up to a limit
    # Powerful numbers: n = a^2 * b^3 where b is square-free
    # Count up to X is ~ c * X^(2/5)
    
    # Try to use a large M; if generation is too slow, fall back
    M = 200_000_000  # 200 million
    
    # Precompute square-free numbers up to M^(1/3) + 2
    max_b = int(M ** (1.0 / 3.0)) + 2  # ~585 for M=200M
    
    # Sieve for smallest prime factor to check square-free quickly
    # For numbers up to max_b, we just need to check if any prime squared divides them
    # Simpler: for each b, check if it's square-free by trial division up to sqrt(b)
    
    # Precompute primes up to sqrt(max_b) for efficiency
    limit_sqrt = isqrt(max_b) + 1
    is_prime = [True] * (limit_sqrt + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit_sqrt) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_sqrt + 1, i):
                is_prime[j] = False
    small_primes = [i for i in range(2, limit_sqrt + 1) if is_prime[i]]
    
    def is_square_free(n):
        temp = n
        for p in small_primes:
            if p * p > temp:
                break
            if temp % (p * p) == 0:
                return False
            while temp % p == 0:
                temp //= p
        # If temp > 1, it's a prime factor with exponent 1 (square-free ok)
        # But we need to check if temp itself is a perfect square
        r = isqrt(temp)
        if r * r == temp and temp > 1:
            return False
        return True
    
    powerful = set()
    
    for b in range(1, max_b + 1):
        if not is_square_free(b):
            continue
        b_cubed = b ** 3
        if b_cubed > M:
            break
        max_a = isqrt(M // b_cubed)
        for a in range(1, max_a + 1):
            powerful.add(a * a * b_cubed)
    
    if time.time() > deadline - 30:
        # Fall back to smaller M
        M = 50_000_000
        max_b = int(M ** (1.0 / 3.0)) + 2
        powerful = set()
        for b in range(1, max_b + 1):
            if not is_square_free(b):
                continue
            b_cubed = b ** 3
            if b_cubed > M:
                break
            max_a = isqrt(M // b_cubed)
            for a in range(1, max_a + 1):
                powerful.add(a * a * b_cubed)
    
    powerful_list = sorted(powerful)
    powerful_set = powerful
    K = len(powerful_list)
    
    if time.time() > deadline - 10:
        # Emergency fallback: return a simple result
        # Find best N from a small subset
        subset = powerful_list[:500]
        counts = defaultdict(int)
        for i in range(len(subset)):
            for j in range(i, len(subset)):
                N = subset[i] + subset[j]
                counts[N] += 1
        best_N = max(counts, key=counts.get) if counts else 2
        pairs = []
        for a in subset:
            if 2 * a > best_N:
                break
            b = best_N - a
            if b in powerful_set and a <= b:
                pairs.append([a, b])
        return {'N': best_N, 'pairs': pairs}
    
    # Step 2: Count pairs using vectorized approach
    # Convert to numpy array
    pw = np.array(powerful_list, dtype=np.int64)
    
    # For each i, compute N = pw[i] + pw[j] for j >= i
    # We want to count occurrences of each N
    
    # Use a dictionary for counting, but process in batches
    # K is around 2000-4000 for M=200M, so K^2/2 is 2M-8M operations
    # Let's do it with numpy for speed
    
    best_N = 0
    best_count = 0
    counts = defaultdict(int)
    
    # Process in chunks to keep memory manageable
    chunk_size = 500
    for i_start in range(0, K, chunk_size):
        i_end = min(i_start + chunk_size, K)
        # For each i in [i_start, i_end), compute pw[i] + pw[j] for j in [i, K)
        for i in range(i_start, i_end):
            a = pw[i]
            # j ranges from i to K-1
            sums = pw[i:] + a
            # Only count N <= 2*M (which is always true since both <= M)
            # Count occurrences
            # Use np.unique for counting
            unique_vals, unique_counts = np.unique(sums, return_counts=True)
            for val, cnt in zip(unique_vals, unique_counts):
                N_val = int(val)
                counts[N_val] += int(cnt)
                if counts[N_val] > best_count:
                    best_count = counts[N_val]
                    best_N = N_val
        
        if time.time() > deadline - 15:
            break
    
    if best_count == 0:
        # Fallback
        best_N = 2
        best_count = 1
    
    # Rebuild pairs for best_N
    pairs = []
    for a in powerful_list:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in powerful_set and a <= b:
            pairs.append([a, b])
    
    return {'N': best_N, 'pairs': pairs}
