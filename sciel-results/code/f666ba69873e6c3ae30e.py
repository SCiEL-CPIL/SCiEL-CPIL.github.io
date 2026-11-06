import math

def construct():
    N = 1000000
    LIMIT = N + 1
    
    # Sieve for primes up to sqrt(LIMIT)
    sqrt_limit = int(math.isqrt(LIMIT))
    is_prime = [True] * (sqrt_limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(sqrt_limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, sqrt_limit + 1, i):
                is_prime[j] = False
                
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # Initialize powerful part array
    pp = [1] * (LIMIT + 1)
    
    # For each prime, update pp for multiples of p^k
    for p in primes:
        p_pow = p * p
        k = 2
        while p_pow <= LIMIT:
            next_pow = p_pow * p
            # Iterate multiples of p_pow
            # If m is a multiple of p_pow but NOT p^(k+1), then exponent of p in m is exactly k
            # Wait, is it exactly k?
            # If m is divisible by p^k and not p^(k+1), then v_p(m) = k.
            # So the powerful part contribution is p^k.
            # If m is divisible by p^(k+1), we skip here, and it will be handled in the next iteration (k+1) or higher.
            # Yes, this partitions the multiples of p^2 into those with v_p(m)=2, v_p(m)=3, etc.
            
            for m in range(p_pow, LIMIT + 1, p_pow):
                if m % next_pow != 0:
                    pp[m] *= p_pow
            
            p_pow = next_pow
            k += 1
            
    # Find best n
    best_n = 10000
    best_val = pp[10000] * pp[10001]
    
    for n in range(10001, N + 1):
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
            
    return best_n
