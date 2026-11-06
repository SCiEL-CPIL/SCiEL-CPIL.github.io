import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 85.0

    # 1. Precompute largest prime factor for all numbers up to N
    # Using a linear-time sieve approach for O(N) complexity
    lpf = [0] * (N + 1)
    lpf[1] = 1
    primes = []
    
    # Linear sieve for largest prime factor
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is prime
            lpf[i] = i
            primes.append(i)
        for p in primes:
            if p > lpf[i] or i * p > N:
                break
            lpf[i * p] = p

    # 2. Search for the optimal split
    # We want to minimize max(lpf[a], lpf[N-a]) for a in [1, N-1]
    best_val = float('inf')
    best_a = 1

    # We can optimize the scan by noting symmetry: max(P(a), P(b)) = max(P(b), P(a))
    # So we only need to check a up to N//2
    half = N // 2
    for a in range(1, half + 1):
        b = N - a
        va = lpf[a]
        vb = lpf[b]
        val = va if va > vb else vb
        
        if val < best_val:
            best_val = val
            best_a = a
            
        # Check time
        if time.time() > deadline:
            break

    return [best_a, N - best_a]
