import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 10_000_000
    
    # Compute sigma(n) for n in [1, N] using NumPy sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
    
    if time.time() > deadline - 15:
        return [11, 6]
    
    # Search for best pair: minimize |2a - 3b| / (2b)
    # Equivalently, minimize |2a - 3b| / b for fixed b, or |2a-3b| for small b
    # For a given b, a ≈ 1.5b, so 2a ≈ 3b, |2a-3b| is small
    # a = (3b ± k) / 2 where k = |2a - 3b|
    
    best_a = 11
    best_b = 6
    best_diff = 4  # |2*11 - 3*6| = 4, error = 4/(2*6) = 1/3
    
    max_k = 100  # Check |2a-3b| up to 100
    
    for b in range(N, 1, -1):
        if b % 5000 == 0 and time.time() > deadline - 5:
            break
        
        sb = int(sigma[b])
        
        for k in range(1, max_k + 1):
            # a1 = (3b + k) / 2
            if (3 * b + k) % 2 == 0:
                a = (3 * b + k) // 2
                if 1 <= a <= N and a != b:
                    if int(sigma[a]) == sb:
                        # Error = k / (2b)
                        # Compare: k/b < best_diff/best_b
                        if k * best_b < best_diff * b:
                            best_diff = k
                            best_a = a
                            best_b = b
                            if best_diff == 1:
                                return [int(best_a), int(best_b)]
            
            # a2 = (3b - k) / 2
            if (3 * b - k) % 2 == 0:
                a = (3 * b - k) // 2
                if 1 <= a <= N and a != b:
                    if int(sigma[a]) == sb:
                        if k * best_b < best_diff * b:
                            best_diff = k
                            best_a = a
                            best_b = b
                            if best_diff == 1:
                                return [int(best_a), int(best_b)]
    
    return [int(best_a), int(best_b)]
