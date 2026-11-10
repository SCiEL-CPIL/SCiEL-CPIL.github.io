import math
import time

def largest_prime_factor(n):
    if n == 1:
        return 1
    max_factor = 1
    # Handle factor 2
    while n % 2 == 0:
        max_factor = 2
        n //= 2
    
    # Check odd factors from 3 to sqrt(n)
    f = 3
    while f * f <= n:
        while n % f == 0:
            max_factor = f
            n //= f
        f += 2
    
    if n > 1:
        max_factor = n
    return max_factor

def construct():
    N = 123457
    best_score = float('inf')
    best_pair = [1, N - 1]
    
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Precompute largest prime factor for all numbers up to N
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (N + 1)
    lpf[1] = 1
    
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    # Iterate through all possible a
    # We can do this very fast now
    for a in range(1, N // 2 + 1):
        if time.time() > deadline:
            break
            
        pa = lpf[a]
        if pa >= best_score:
            continue
            
        b = N - a
        pb = lpf[b]
        current_score = max(pa, pb)
        
        if current_score < best_score:
            best_score = current_score
            best_pair = [a, b]
            
            # If we find a score of 2, it's likely optimal or near optimal?
            # P(n)=1 only for n=1.
            # If a=1, P(a)=1. b=N-1. P(N-1) must be small.
            # If we find max=2, it means both are powers of 2.
            # 2^k + 2^m = N.
            # This is a strong constraint.
            # If we find max=3, both are 3-smooth.
            
    return best_pair
