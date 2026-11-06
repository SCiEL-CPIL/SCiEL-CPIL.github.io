"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 1000000
    
    # Build SPF table
    spf = np.arange(N + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part using sieve-like accumulation
    # pp[i] = product of p^e for all prime powers p^e || i with e >= 2
    pp = np.ones(N + 1, dtype=np.int64)
    
    # For each prime p, for each power p^e with e >= 2, multiply pp[j] by p^e
    # for all j that are divisible by p^e
    for p in range(2, N + 1):
        if spf[p] != p:
            continue  # not a prime
        
        # Compute p^2, p^3, ... and multiply into pp
        power = p * p
        while power <= N:
            # Multiply pp[j] by p^e (which is power) for all j divisible by power
            # But we need to be careful: if j is divisible by p^e but not p^(e+1),
            # then p^e is the full prime power. If j is divisible by p^(e+1),
            # then p^(e+1) is the full prime power, not p^e.
            # So we should only multiply by p^e for j divisible by p^e but NOT p^(e+1).
            # However, a simpler approach: for each j, the powerful part includes
            # p^e where e is the exact exponent of p in j. So we need to find for
            # each j, the highest power of p dividing j, and include that if >= 2.
            
            # Let's use a different approach: for each prime p, find the exact power
            # p^e || j for each j, and if e >= 2, multiply pp[j] by p^e.
            # This can be done by iterating through multiples of p^2, p^3, etc.
            # but only multiplying by p^e (not p^(e-1)) for those j where p^e || j.
            
            # Simpler: iterate through all j, but that's slow. Let's stick with
            # the original per-number factorization but vectorize where possible.
            break  # We'll use a different approach below
    
    # Use per-number factorization with SPF, but optimize the inner loop
    pp_list = [1] * (N + 1)
    for i in range(2, N + 1):
        temp = i
        prod = 1
        while temp > 1:
            p = spf[temp]
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            if e >= 2:
                prod *= p ** e
        pp_list[i] = prod
    
    pp = np.array(pp_list, dtype=np.int64)
    
    # Find max product pp[n] * pp[n+1] for n in [10000, 1000000]
    # Vectorize the product computation
    products = pp[10000:N] * pp[10001:N+1]
    max_idx = np.argmax(products)
    best_n = 10000 + max_idx
    
    return int(best_n)

# EVOLVE-BLOCK-END
