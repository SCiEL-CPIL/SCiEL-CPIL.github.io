import math
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    N = 1000000
    min_n = 10000
    max_n = N - 1  # n+1 must be <= 1000000
    
    # Step 1: Generate primes up to sqrt(N) using Sieve of Eratosthenes
    limit = int(math.isqrt(N)) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Step 2: Compute powerful part for all numbers up to N
    # pp[i] will store the powerful part of i
    pp = [1] * (N + 1)
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        
        # Iterate through multiples of p^2
        # For each multiple m, find the highest power of p dividing m
        # m = p^k * rest, where rest is not divisible by p
        # We want to multiply pp[m] by p^k (since k >= 2)
        
        # Optimization: Precompute powers of p
        # But we can just divide m by p repeatedly.
        # Actually, we can iterate k from 2 upwards? No, m is fixed.
        
        # Let's just loop m = p2, p2+p, p2+2p, ...
        # Wait, step is p, not p2?
        # If m is a multiple of p^2, it is also a multiple of p.
        # We only care about m that are multiples of p^2.
        # So step is p^2? No.
        # Multiples of p^2 are p^2, 2p^2, 3p^2...
        # But what about 3p^2? It is a multiple of p^2.
        # What about p^2 * (p+1)? It is a multiple of p^2.
        # So yes, we iterate m in range(p2, N+1, p2)?
        # No. 3*p^2 is a multiple of p^2.
        # But is every multiple of p^2 covered by range(p2, N+1, p2)?
        # Yes.
        # However, consider m = p^2 * 3.
        # v_p(m) = 2.
        # Consider m = p^3.
        # v_p(m) = 3.
        # p^3 is a multiple of p^2. It is in the range.
        
        # So, for m in range(p2, N+1, p2):
        #   We know p^2 divides m.
        #   We need to find the exact power p^k.
        
        # Actually, iterating with step p is safer to catch all multiples, 
        # but we only update if p^2 divides m.
        # Iterating with step p^2 is more efficient.
        
        for m in range(p2, N + 1, p2):
            # Find highest power of p dividing m
            # m is divisible by p^2
            temp = m
            power = 1
            # We know it's divisible by p at least twice
            # So we can start power at p*p?
            # No, let's just divide.
            # Since m is a multiple of p^2, we can divide by p twice immediately?
            # No, let's just loop.
            # To speed up, we can do:
            # power = p*p
            # temp = m // (p*p)
            # while temp % p == 0:
            #    temp //= p
            #    power *= p
            
            # This is faster.
            base_power = p2
            temp = m // p2
            power = base_power
            while temp % p == 0:
                temp //= p
                power *= p
            
            pp[m] *= power
    
    # Step 3: Scan for max product
    best_n = min_n
    best_score = 0
    
    # Check time
    if time.time() - start_time > deadline - 5:
         # If we ran out of time during sieve (unlikely), return a heuristic
         # But sieve is fast.
         pass
         
    for n in range(min_n, max_n + 1):
        if time.time() - start_time > deadline:
            break
        score = pp[n] * pp[n+1]
        if score > best_score:
            best_score = score
            best_n = n
            
    return best_n
