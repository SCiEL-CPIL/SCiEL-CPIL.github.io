"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline for safety margin
    
    MAX_N = 1000000
    MAX_VAL = MAX_N + 1
    
    # Compute powerful part for each number
    # powerful_part[n] = product of p^e where e >= 2 in factorization of n
    powerful_part = [1] * (MAX_VAL + 1)
    
    # For each prime p, for each power p^2, p^3, ... <= MAX_VAL
    # multiply p^(e-2) into powerful_part for multiples of p^e but not p^(e+1)
    # Actually, let's think differently:
    # For each number n, powerful_part(n) = product of p^e where p^e || n and e >= 2
    
    # Better approach: for each prime p, find all numbers divisible by p^2
    # For such numbers, the contribution to powerful part from p is p^e where p^e || n
    
    # Let's use a sieve-like approach
    # For each prime p >= 2:
    #   For each power p^k (k>=2) where p^k <= MAX_VAL:
    #     For each multiple m of p^k:
    #       If m is NOT divisible by p^(k+1), then p^k contributes to powerful_part[m]
    #       But we need the FULL p^e, not just p^k
    
    # Actually, simpler: for each number n, factor it and compute powerful part.
    # But that's O(n log n) which might be slow for n up to 10^6.
    
    # Let me use a sieve approach:
    # powerful_part[n] = 1 initially
    # For each prime p:
    #   p2 = p*p
    #   For each multiple m of p2:
    #     Count how many times p divides m: let e = v_p(m)
    #     Then p^e contributes to powerful_part[m]
    
    # Efficient way:
    # For each prime p:
    #   For e = 2, 3, 4, ... while p^e <= MAX_VAL:
    #     For each m divisible by p^e but not p^(e+1):
    #       powerful_part[m] *= p^e
    
    # Even simpler: 
    # For each prime p:
    #   For each m divisible by p^2:
    #     e = v_p(m)  (can compute by dividing)
    #     powerful_part[m] *= p^e
    
    # Let me just do trial division with smallest prime factor sieve
    
    # Step 1: Compute smallest prime factor (SPF) for all numbers up to MAX_VAL
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(math.isqrt(MAX_VAL)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Step 2: For each number, compute powerful part using SPF
    for n in range(2, MAX_VAL + 1):
        m = n
        pp = 1
        while m > 1:
            p = spf[m]
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            if e >= 2:
                pp *= p ** e
        powerful_part[n] = pp
    
    # Check time
    if time.time() > deadline:
        return 10000
    
    # Step 3: Find n in [10000, MAX_N] that maximizes powerful_part[n] * powerful_part[n+1]
    best_n = 10000
    best_val = powerful_part[10000] * powerful_part[10001]
    
    for n in range(10000, MAX_N):
        val = powerful_part[n] * powerful_part[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
