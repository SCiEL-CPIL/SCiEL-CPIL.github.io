def construct():
    N = 256
    # Generate primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    S = set(primes)
    
    # Add prime squares
    for p in primes:
        sq = p * p
        if sq <= N:
            S.add(sq)
            
    return [int(x) for x in S]
