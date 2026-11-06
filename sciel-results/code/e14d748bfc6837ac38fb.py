"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    N = 1000000
    LO = 10000
    
    # Segmented-sieve style: compute powerful part for every integer in [1, N]
    # powerful_part[n] = product of p^e for each prime p where e >= 2 in factorization of n
    
    # Initialize array of 1s
    pp = [1] * (N + 1)
    
    # For each prime p, for each power p^2, p^3, ... <= N,
    #   for each multiple m of p^k (where k>=2), multiply pp[m] by p^(highest exponent of p in m)
    # Actually, a cleaner approach:
    # For each prime p, for each multiple of p, determine the exact exponent e of p in that multiple.
    # If e >= 2, multiply pp[multiple] by p^e.
    
    # First, find all primes up to N using a simple sieve
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # For each prime, process its multiples
    for p in primes:
        # Precompute powers of p
        p2 = p * p
        if p2 > N:
            break
        
        # For each multiple of p, compute the exponent of p in it
        # We iterate through multiples of p and track the power
        # Efficient approach: for each power p^k (k>=2), for each multiple m of p^k,
        #   we need to add p^(e) where e is the full exponent. 
        # Better: for each multiple m of p, compute v_p(m) and if >= 2, multiply by p^v_p(m)
        
        # To avoid per-number factorization, use the following:
        # For p^2, p^3, ..., p^k <= N:
        #   For each multiple of p^i (i from 2 to k), but not of p^(i+1), the exponent is exactly i.
        #   So we can do: for i from 2 while p^i <= N:
        #     For each multiple m of p^i that is NOT a multiple of p^(i+1):
        #       pp[m] *= p^i
        
        # Even simpler: just iterate multiples of p and compute exponent directly
        # Since we're doing this for each prime separately, and the total work is sum over primes of N/p,
        # this is O(N log log N) which is fine.
        
        # For each multiple of p:
        mult = p
        while mult <= N:
            # Compute v_p(mult)
            temp = mult
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            if exp >= 2:
                pp[mult] *= (p ** exp)
            mult += p
        
        if time.time() > deadline:
            break
    
    # Now find the best n in [LO, N-1] maximizing pp[n] * pp[n+1]
    best_n = LO
    best_val = pp[LO] * pp[LO + 1]
    
    for n in range(LO, N):
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
