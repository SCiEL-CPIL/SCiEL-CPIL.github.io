import math

def construct():
    M = 1000000
    limit = M + 1
    
    # Sieve of Eratosthenes to find smallest prime factor
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Extract primes
    primes = [i for i in range(2, limit + 1) if spf[i] == i]
    
    # R[n] will store the product of primes p where v_p(n) == 1
    # P(n) = n / R[n]
    R = [1] * (limit + 1)
    
    for p in primes:
        p_sq = p * p
        # Iterate multiples of p
        # If j is a multiple of p but not p^2, then v_p(j) == 1
        # We only need to consider j up to limit
        for j in range(p, limit + 1, p):
            if j % p_sq != 0:
                R[j] *= p
                
    # Calculate powerful part and find max
    max_val = 0
    best_n = 10000
    
    # We can compute P(n) on the fly or precompute
    # Precomputing P is O(N)
    powerful_part = [0] * (limit + 1)
    for i in range(1, limit + 1):
        powerful_part[i] = i // R[i]
        
    for n in range(10000, M + 1):
        val = powerful_part[n] * powerful_part[n+1]
        if val > max_val:
            max_val = val
            best_n = n
            
    return best_n
