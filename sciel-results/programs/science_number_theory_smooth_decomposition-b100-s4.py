import time

def construct():
    N = 123457
    deadline = time.time() + 85  # safety margin
    
    # Sieve to compute largest prime factor for all numbers 1..N
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for i in range(2, N + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    # Find the best (a, b) with a+b=N minimizing max(lpf[a], lpf[b])
    best_max = N + 1
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        b = N - a
        m = lpf[a] if lpf[a] > lpf[b] else lpf[b]
        if m < best_max:
            best_max = m
            best_a = a
            best_b = b
            if best_max == 2:
                # Can't do better than 2 (since P(1)=1, but 1+N-1... wait)
                # Actually P(1)=1, so if a=1, b=N, max(1, P(N)). 
                # If both are powers of 2, max would be 2. But 2^k + 2^m = N?
                # Unlikely. But 2 is the smallest possible prime factor > 1.
                # If we find max=2, that means both are powers of 2 (or 1).
                # 1 + (N-1): P(1)=1, P(N-1)=? N-1=123456. 
                # Let's just break if we hit the theoretical minimum.
                # Theoretical min: if a=1, b=N-1, max(1, P(N-1)).
                # If N-1 is a power of 2, then P(N-1)=2, so max=2.
                # 123456 = 2^4 * 7716 = 16 * 7716. 7716/4=1929, odd. So not power of 2.
                # So 2 is likely not achievable. But let's not break early, just continue.
                pass
    
    if time.time() > deadline:
        return [best_a, best_b]
    
    return [best_a, best_b]
