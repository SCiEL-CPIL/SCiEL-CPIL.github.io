"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    N = 1000000
    LO = 10000
    
    # Compute powerful part for each number in [LO, N]
    # powerful_part(k) = product of p^e for each prime p^e || k where e >= 2
    # Equivalently: k / (squarefree kernel of k) ... not quite.
    # Let k = product p_i^{e_i}. powerful_part = product_{e_i>=2} p_i^{e_i}.
    # So powerful_part(k) = k / product_{e_i=1} p_i.
    
    # We'll compute for each k the "squarefree part" (product of primes with exponent exactly 1)
    # then powerful_part(k) = k / squarefree_part(k).
    
    # Sieve approach: for each prime p, find all multiples.
    # For each k, we need to know which primes divide it to exactly the first power.
    
    # Alternative: compute rad(k) and the "square part".
    # Let's compute: for each k, factor partially.
    
    # Better approach: 
    # 1. Compute smallest prime factor (SPF) for all numbers up to N.
    # 2. For each k, factor it using SPF to get exponents.
    # 3. Compute powerful_part(k).
    
    # Step 1: SPF sieve
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Step 2 & 3: For each k in [LO, N], compute powerful_part
    # We only need k and k+1 both in range, so k in [LO, N-1]
    # powerful_part[k] for k in [LO, N]
    
    pow_part = [1] * (N + 1)
    pow_part[0] = 0
    pow_part[1] = 1
    
    for k in range(2, N + 1):
        n = k
        pp = 1
        while n > 1:
            p = spf[n]
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            if e >= 2:
                pp *= (p ** e)
        pow_part[k] = pp
        if time.time() > deadline - 5:
            break
    
    # Now find max of pow_part[k] * pow_part[k+1] for k in [LO, N-1]
    best_val = 0
    best_k = LO
    for k in range(LO, N):
        val = pow_part[k] * pow_part[k + 1]
        if val > best_val:
            best_val = val
            best_k = k
    
    return best_k
# EVOLVE-BLOCK-END
