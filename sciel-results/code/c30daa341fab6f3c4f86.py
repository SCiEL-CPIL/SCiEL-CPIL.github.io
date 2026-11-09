def construct():
    import time
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s kill
    
    limit = 1000000
    N = limit + 1
    
    # Precompute primes up to sqrt(N) for trial division
    import math
    sqrt_n = int(math.isqrt(N)) + 1
    is_prime = [True] * (sqrt_n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(sqrt_n)) + 1):
        if is_prime[i]:
            for j in range(i*i, sqrt_n + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, sqrt_n + 1) if is_prime[i]]
    
    def largest_prime_factor(num):
        """Compute largest prime factor via trial division by precomputed primes."""
        if num <= 1:
            return 1
        lpf = 1
        temp = num
        for p in primes:
            if p * p > temp:
                break
            while temp % p == 0:
                lpf = p
                temp //= p
        if temp > 1:
            lpf = temp
        return lpf
    
    best_val = float('inf')
    best_n = 10000
    
    # Search for n in [10000, 1000000]
    for n in range(10000, limit + 1):
        if time.time() >= deadline:
            break
        v1 = largest_prime_factor(n)
        v2 = largest_prime_factor(n + 1)
        v = v1 if v1 > v2 else v2
        if v < best_val:
            best_val = v
            best_n = n
    
    return best_n
