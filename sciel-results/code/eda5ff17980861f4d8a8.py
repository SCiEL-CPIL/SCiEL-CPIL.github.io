import math
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    def generate_powerful(limit):
        """
        Generates powerful numbers up to 'limit'.
        A powerful number is one where every prime factor p divides n^2.
        Equivalently, n = a^2 * b^3 where b is square-free.
        """
        powerful = set()
        max_b = int(limit ** (1.0/3.0)) + 2
        
        # Precompute primes up to sqrt(max_b) to check square-free
        # max_b <= (10^12)^(1/3) = 10^4. sqrt(10^4) = 100.
        # Actually, we need to check if b is square-free.
        # b is square-free if it is not divisible by any prime square p^2.
        
        def get_prime_squares(limit):
            if limit < 2:
                return []
            is_prime = [True] * (limit + 1)
            is_prime[0] = False
            if limit >= 1:
                is_prime[1] = False
            for i in range(2, int(limit ** 0.5) + 1):
                if is_prime[i]:
                    for j in range(i*i, limit + 1, i):
                        is_prime[j] = False
            squares = [i*i for i in range(2, limit + 1) if is_prime[i]]
            return squares

        max_b_sqrt = int(max_b ** 0.5) + 2
        prime_squares = get_prime_squares(max_b_sqrt)
        
        def is_square_free(n):
            if n < 2:
                return True
            for ps in prime_squares:
                if ps > n:
                    break
                if n % ps == 0:
                    return False
            return True

        for b in range(1, max_b + 1):
            if not is_square_free(b):
                continue
            b3 = b ** 3
            if b3 > limit:
                break
            # a^2 * b^3 <= limit => a^2 <= limit / b^3 => a <= sqrt(limit / b^3)
            max_a = int(math.isqrt(limit // b3))
            for a in range(1, max_a + 1):
                val = a * a * b3
                if val <= limit:
                    powerful.add(val)
        
        return sorted(powerful)

    # Generate powerful numbers up to 10^12
    # M(10^12) is approx 10^6.
    # Generation time is acceptable.
    L = 10**12
    P = generate_powerful(L)
    P_set = set(P)
    n_P = len(P)
    
    if time.time() > deadline - 30:
        # Fallback to smaller search if generation took too long (unlikely)
        return {'N': 2, 'pairs': [[1, 1]]}

    # Strategy: Sample candidate N values from the upper range of sums.
    # We want to find N with many representations.
    # Expected representations scale with sqrt(N).
    # So we focus on large N.
    
    # We will sample K candidates.
    # To ensure we cover a good range, we pick random pairs (a, b) from P
    # but bias them towards the larger values.
    
    # Let's pick candidates from the top 50% of the list to ensure N is large.
    # Indices from n_P//2 to n_P-1.
    
    K = 200  # Number of candidates to verify
    candidates = set()
    
    # Ensure we have enough elements in the top half
    start_idx = max(0, n_P // 2)
    top_P = P[start_idx:]
    
    if not top_P:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Random sampling of pairs from top_P
    # To avoid duplicates and ensure diversity, we can just pick random indices.
    # We also add a few deterministic "high" sums, e.g., largest + largest, etc.
    
    # Add some deterministic high-value candidates
    for i in range(10):
        a = top_P[-1 - i] if len(top_P) > i else top_P[-1]
        b = top_P[-1 - i] if len(top_P) > i else top_P[-1]
        candidates.add(a + b)
        
    for _ in range(K):
        if time.time() > deadline - 10:
            break
        idx_a = random.randint(0, len(top_P) - 1)
        idx_b = random.randint(0, len(top_P) - 1)
        a = top_P[idx_a]
        b = top_P[idx_b]
        candidates.add(a + b)
        
    # Verify each candidate
    best_N = 0
    best_count = 0
    best_pairs = []
    
    for N in candidates:
        if time.time() > deadline - 5:
            break
            
        count = 0
        pairs = []
        # Iterate through P to find a such that N-a is in P
        # Optimization: only need a <= N/2 to avoid double counting
        half_N = N // 2
        
        # Binary search for the index where P[i] > half_N
        # Or just iterate and break.
        # Since P is sorted, we can break early.
        
        for a in P:
            if a > half_N:
                break
            b = N - a
            if b in P_set:
                count += 1
                pairs.append([a, b])
                
        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = pairs
            
    if best_N == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_N, 'pairs': best_pairs}
