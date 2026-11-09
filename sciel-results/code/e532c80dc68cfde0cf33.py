import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    N_MIN = 10000
    N_MAX = 1000000
    LIMIT = N_MAX + 1
    
    # Sieve to find primes up to sqrt(LIMIT)
    sqrt_limit = int(LIMIT ** 0.5) + 1
    is_prime = [True] * (sqrt_limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(sqrt_limit ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, sqrt_limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, sqrt_limit + 1) if is_prime[i]]
    
    # Powerful part array
    pp = [1] * (LIMIT + 1)
    
    # For each prime, compute its contribution to pp[m] for all m
    for p in primes:
        p2 = p * p
        if p2 > LIMIT:
            break
        # Precompute p^e for e >= 2 up to LIMIT
        powers = []
        pk = p2
        e = 2
        while pk <= LIMIT:
            powers.append((pk, e))
            pk *= p
            e += 1
        
        # For each multiple of p2, determine the exact power
        for m in range(p2, LIMIT + 1, p):
            # Find the highest power of p dividing m
            # m is divisible by p2, so e >= 2
            e_count = 2
            temp = m // p2
            while temp % p == 0:
                e_count += 1
                temp //= p
            pp[m] *= p ** e_count
    
    # Scan for best n in [N_MIN, N_MAX]
    best_n = N_MIN
    best_val = pp[N_MIN] * pp[N_MIN + 1]
    
    for n in range(N_MIN, N_MAX):
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        
        if time.time() - start_time > DEADLINE:
            break
    
    return best_n
