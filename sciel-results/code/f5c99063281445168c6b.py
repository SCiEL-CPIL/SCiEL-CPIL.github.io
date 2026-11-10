import numpy as np
import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = 45.0  # Safety margin (evaluator kills at 90s)
    
    # 1. Sieve of Eratosthenes to find primes up to N-1
    limit = N - 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = False
            
    primes = np.where(is_prime)[0]
    
    # 2. Compute Largest Prime Factor (LPF) for all numbers up to N-1
    lpf = np.ones(limit + 1, dtype=np.int64)
    
    for p in primes:
        lpf[p::p] = p
        
    if time.time() - start_time > deadline:
        return (1, N - 1)

    # 3. Find the split a + b = N that minimizes max(P(a), P(b))
    # Vectorized exhaustive search
    a_arr = np.arange(1, N, dtype=np.int64)
    b_arr = N - a_arr
    
    cost = np.maximum(lpf[a_arr], lpf[b_arr])
    
    min_cost = np.min(cost)
    min_idx = np.argmin(cost)
    
    best_a = int(a_arr[min_idx])
    best_b = int(b_arr[min_idx])
    
    # Verify correctness
    assert best_a > 0 and best_b > 0 and best_a + best_b == N, f"Invalid: {best_a}+{best_b}={best_a+best_b}"
    
    return (best_a, best_b)
