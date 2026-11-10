def construct():
    N = 123457
    
    # 1. Sieve for Smallest Prime Factors (SPF)
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute Largest Prime Factor P(n) for all n <= N
    # P(1) = 1 by definition
    P = [1] * (N + 1)
    for i in range(2, N + 1):
        # Factorize i using spf
        x = i
        max_pf = 1
        while x > 1:
            p = spf[x]
            max_pf = p
            while x % p == 0:
                x //= p
        P[i] = max_pf
    
    # 3. Find a, b with a+b=N minimizing max(P(a), P(b))
    min_cost = float('inf')
    best_a = 1
    best_b = N - 1
    
    # We only need to check a up to N//2 to avoid duplicate pairs, 
    # but since we want specific a,b, checking all is fine and simple.
    # Actually, a and b are symmetric in cost, so checking a in [1, N//2] is sufficient 
    # to find the optimal pair values, but we need to return the specific pair.
    # Let's just iterate all a from 1 to N-1.
    for a in range(1, N):
        b = N - a
        cost = max(P[a], P[b])
        if cost < min_cost:
            min_cost = cost
            best_a = a
            best_b = b
            # Early exit if we hit the theoretical minimum? 
            # The minimum possible max prime factor is at least 2 (unless one is 1).
            # If P(a)=1, a=1. P(b)=P(N-1).
            # If we find cost=2, that's very good. Can we do better? No, primes are >=2.
            # So if min_cost == 2, we can stop.
            if min_cost == 2:
                break
    
    return [best_a, best_b]
