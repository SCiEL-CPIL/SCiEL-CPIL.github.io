import time
import math

def construct():
    start_time = time.time()
    # 90s limit, use 85s as deadline
    deadline = start_time + 85.0
    
    # We need P(n) and P(n+1) for n up to 1,000,000.
    # So we need to compute P up to 1,000,001.
    N_MAX = 1000001
    MIN_N = 10000
    
    # 1. Sieve of Smallest Prime Factor (SPF)
    spf = list(range(N_MAX + 1))
    # spf[0] and spf[1] are not used for factorization in the loop starting from 2
    # But we should ensure spf[1] is handled or ignored.
    # For i in range(2, sqrt(N_MAX)):
    for i in range(2, int(math.isqrt(N_MAX)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N_MAX + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute Powerful Part P[n] for all n in [1, N_MAX]
    P = [1] * (N_MAX + 1)
    # P[1] = 1
    for n in range(2, N_MAX + 1):
        x = n
        p_part = 1
        while x > 1:
            prime = spf[x]
            exp = 0
            while x % prime == 0:
                x //= prime
                exp += 1
            if exp >= 2:
                p_part *= (prime ** exp)
        P[n] = p_part
        
        # Periodic time check
        if n % 500000 == 0:
            if time.time() > deadline:
                # If we run out of time, we can't compute all P.
                # But this loop is fast. If it happens, we break and use what we have.
                # However, we need P[n+1] for the final loop.
                # If we break early, we might have incomplete P array.
                # We'll just break and hope we finished.
                break

    # 3. Find best n
    max_score = 0
    best_n = MIN_N
    
    # n ranges from 10000 to 1000000
    # We need P[n] and P[n+1].
    # If we broke early in step 2, P might be 1 for uncomputed values.
    # This is a safe fallback.
    
    for n in range(MIN_N, 1000001):
        if time.time() > deadline:
            break
        score = P[n] * P[n + 1]
        if score > max_score:
            max_score = score
            best_n = n
            
    return best_n
