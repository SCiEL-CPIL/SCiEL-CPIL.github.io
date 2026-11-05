import math

def construct():
    N = 1000000
    MAX_VAL = N + 1
    
    # Sieve to find Largest Prime Factor (LPF) for all numbers up to MAX_VAL
    # lpf[i] will store the largest prime factor of i
    lpf = list(range(MAX_VAL + 1))
    
    # Sieve of Eratosthenes variant for LPF
    # Iterate through potential prime factors
    for i in range(2, int(math.isqrt(MAX_VAL)) + 1):
        if lpf[i] == i: # i is prime
            for j in range(i*i, MAX_VAL + 1, i):
                lpf[j] = i # Since i is increasing, this updates to larger primes eventually? 
                # Wait. 
                # If I iterate i from 2 upwards:
                # For j=12:
                # i=2: lpf[12]=2
                # i=3: lpf[12]=3
                # i=5: no
                # ...
                # So yes, if I iterate primes in increasing order, the last assignment is the largest prime factor.
                # But I only loop i up to sqrt(MAX_VAL).
                # What if the largest prime factor is > sqrt(MAX_VAL)?
                # If j is prime, lpf[j] remains j. Correct.
                # If j = p * q where p < q and q > sqrt(MAX_VAL), then p < sqrt(MAX_VAL).
                # The loop will hit p. lpf[j] becomes p.
                # It will not hit q because q > sqrt(MAX_VAL).
                # So lpf[j] will be p, which is NOT the largest prime factor (q is larger).
                # So this simple sieve only gives the SMALLEST prime factor if we stop at sqrt.
                # Or rather, it gives the smallest prime factor found in the sieve range.
                
    # Correction:
    # The standard SPF sieve gives the smallest prime factor.
    # To get LPF, we can just use the fact that if lpf[j] is set by a small prime, we might miss the large one.
    # However, for any composite number $j$, its largest prime factor is either $j$ itself (if prime) or it has a factor $p \le \sqrt{j}$.
    # Actually, simpler approach:
    # 1. Sieve for Smallest Prime Factor (SPF).
    # 2. For each number, divide by SPF repeatedly to find the largest prime factor? That's $O(N \log N)$ or similar, still fast enough for $10^6$.
    # 3. Or, just do the sieve correctly for LPF.
    
    # Let's restart the sieve logic for LPF.
    # We want lpf[i] to be the largest prime factor of i.
    # Initialize lpf[i] = i.
    # For i from 2 to MAX_VAL:
    #   If lpf[i] == i (i is prime):
    #      For j in range(i, MAX_VAL+1, i):
    #          lpf[j] = i
    # This works because we iterate i in increasing order. The last prime i that divides j will be the largest one.
    # Complexity: Sum over primes p <= N of N/p = N * ln(ln(N)). This is very efficient.
    
    lpf = list(range(MAX_VAL + 1))
    for i in range(2, MAX_VAL + 1):
        if lpf[i] == i: # i is prime
            for j in range(i, MAX_VAL + 1, i):
                lpf[j] = i
                
    best_n = 10000
    min_lpf_val = float('inf')
    
    for n in range(10000, N + 1):
        val = max(lpf[n], lpf[n+1])
        if val < min_lpf_val:
            min_lpf_val = val
            best_n = n
            
    return best_n
