import time
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    def time_left():
        return deadline - time.time()
    
    # Generate powerful numbers up to M
    # Powerful number: n = a^2 * b^3 where b is square-free
    # Count of powerful numbers up to X is ~ c * X^(2/5)
    # For M = 100_000_000, count ~ 100 * (100_000_000)^(2/5) ~ 100 * 1000 = 100,000
    # But let's be more precise: the constant is around 0.5-1.0
    # Let's use M = 50_000_000 to be safe on time
    
    M = 50_000_000
    
    # Sieve for square-free numbers up to M^(1/3)
    max_b = int(M ** (1.0 / 3.0)) + 2
    
    # Sieve of Eratosthenes to find primes up to max_b
    is_prime = [True] * (max_b + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(max_b) + 1):
        if is_prime[i]:
            for j in range(i*i, max_b + 1, i):
                is_prime[j] = False
    
    # Find square-free numbers
    # A number is square-free if it's not divisible by p^2 for any prime p
    is_square_free = [True] * (max_b + 1)
    is_square_free[0] = False
    for i in range(2, isqrt(max_b) + 1):
        if is_prime[i]:
            p2 = i * i
            for j in range(p2, max_b + 1, p2):
                is_square_free[j] = False
    
    powerful = []
    for b in range(1, max_b + 1):
        if not is_square_free[b]:
            continue
        b_cubed = b ** 3
        if b_cubed > M:
            break
        max_a = isqrt(M // b_cubed)
        for a in range(1, max_a + 1):
            powerful.append(a * a * b_cubed)
    
    powerful.sort()
    P_set = set(powerful)
    K = len(powerful)
    
    # Now we want to find N that maximizes the number of pairs (a, b) with a + b = N
    # We can use a dictionary to count sums
    # But K can be large, so K^2 pairs might be too many
    # Let's estimate: if M = 50M, K ~ 20,000 (rough estimate)
    # K^2 = 400M, which is too many
    
    # Alternative: use FFT on a smaller scale
    # Or: sample a subset of powerful numbers
    
    # Let's try a different approach: use the fact that we can compute
    # the number of representations using convolution via FFT
    # But limit the size to avoid memory issues
    
    import numpy as np
    
    # Use FFT approach but with a reasonable size
    # We need to compute sums up to 2*M
    # But we can use a smaller M for the FFT and still get good results
    
    M_fft = 10_000_000  # Smaller limit for FFT
    
    # Regenerate powerful numbers up to M_fft
    max_b_fft = int(M_fft ** (1.0 / 3.0)) + 2
    is_square_free_fft = [True] * (max_b_fft + 1)
    is_square_free_fft[0] = False
    for i in range(2, isqrt(max_b_fft) + 1):
        if is_prime[i] and i <= max_b_fft:
            p2 = i * i
            for j in range(p2, max_b_fft + 1, p2):
                is_square_free_fft[j] = False
    
    powerful_fft = []
    for b in range(1, max_b_fft + 1):
        if not is_square_free_fft[b]:
            continue
        b_cubed = b ** 3
        if b_cubed > M_fft:
            break
        max_a = isqrt(M_fft // b_cubed)
        for a in range(1, max_a + 1):
            powerful_fft.append(a * a * b_cubed)
    
    powerful_fft.sort()
    P_fft_set = set(powerful_fft)
    
    # FFT size
    target_size = 2 * M_fft
    fft_size = 1
    while fft_size < target_size:
        fft_size <<= 1
    
    A = np.zeros(fft_size, dtype=np.float64)
    for p in powerful_fft:
        if p < fft_size:
            A[p] = 1.0
    
    F = np.fft.fft(A)
    G = F * F
    C = np.fft.ifft(G)
    Counts = np.rint(C.real).astype(np.int64)
    
    # Find the N with maximum number of pairs
    # D(N) = (Counts[N] + n_eq(N)) / 2
    best_score = 0
    best_N = -1
    
    # Check top candidates
    top_k = 500
    if len(Counts) > top_k:
        valid_counts = Counts[2:2*M_fft+1]
        part_idx = np.argpartition(valid_counts, -top_k)[-top_k:]
        top_vals = valid_counts[part_idx]
        sorted_idx = np.argsort(top_vals)[::-1]
        top_indices = part_idx[sorted_idx] + 2
        
        for idx in top_indices:
            N = int(idx)
            c = int(Counts[N])
            n_eq = 0
            if N % 2 == 0:
                half = N // 2
                if half in P_fft_set:
                    n_eq = 1
            score = (c + n_eq) // 2
            if score > best_score:
                best_score = score
                best_N = N
    
    if best_N == -1:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Reconstruct pairs for best_N
    pairs = []
    limit = best_N // 2
    for a in powerful_fft:
        if a > limit:
            break
        b = best_N - a
        if b in P_fft_set:
            pairs.append([a, b])
    
    return {'N': int(best_N), 'pairs': pairs}
