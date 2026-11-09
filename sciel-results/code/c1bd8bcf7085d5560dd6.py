"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import math

    # Use FFT to count sums efficiently, allowing larger effective search
    # Memory: Next power of 2 for 2*LIMIT.
    # LIMIT = 3*10^7 -> 2*LIMIT = 6*10^7 -> 2^26 = 6.7*10^7.
    # Memory ~ 2GB. Safe.
    LIMIT = 3 * 10**7
    
    # 1. Generate Powerful Numbers
    powerful_set = set()
    max_b = int(LIMIT ** (1/3))
    
    limit_sqrt = int(math.isqrt(max_b)) + 1
    is_prime = [True] * limit_sqrt
    if limit_sqrt > 0:
        is_prime[0] = False
    if limit_sqrt > 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit_sqrt)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_sqrt, i):
                is_prime[j] = False
    primes = [i for i, p in enumerate(is_prime) if p]
    prime_squares = [p*p for p in primes]
    
    def is_square_free(n):
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
        max_a = int(math.isqrt(LIMIT // b3))
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= LIMIT:
                powerful_set.add(val)
                
    P = sorted(powerful_set)
    P_set = powerful_set
    
    # 2. FFT Counting
    size = 1
    while size < 2 * LIMIT:
        size *= 2
        
    A = np.zeros(size, dtype=np.float64)
    for val in P:
        A[val] = 1.0
        
    F = np.fft.fft(A)
    C = np.fft.ifft(F * F).real
    
    # 3. Find Best N
    # We look for max in [2, 2*LIMIT]
    # C is float, values are counts.
    valid_slice = C[2 : 2*LIMIT + 1]
    
    # Find the index of the maximum
    idx_max = np.argmax(valid_slice) + 2
    best_N = int(idx_max)
    
    # Verify top candidates to handle float precision or ties
    candidates = set()
    for d in range(-5, 6):
        n_cand = best_N + d
        if 2 <= n_cand <= 2 * LIMIT:
            candidates.add(n_cand)
            
    best_score = 0
    best_N_final = best_N
    
    for N in candidates:
        ordered_count = int(round(C[N]))
        self_count = 0
        if N % 2 == 0:
            half = N // 2
            if half in P_set:
                self_count = 1
        distinct_pairs = (ordered_count + self_count) // 2
        
        if distinct_pairs > best_score:
            best_score = distinct_pairs
            best_N_final = N
            
    # 4. Retrieve Pairs
    pairs = []
    for a in P:
        b = best_N_final - a
        if a <= b and b in P_set:
            pairs.append([a, b])
            
    pairs.sort()
    
    return {'N': best_N_final, 'pairs': pairs}
# EVOLVE-BLOCK-END
