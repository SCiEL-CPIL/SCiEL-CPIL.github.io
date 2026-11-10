import math

def construct():
    # Upper bound for n+1
    max_n = 1000000
    limit = max_n + 1
    
    # Sieve for Largest Prime Factor (LPF)
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (limit + 1)
    
    # We only need to iterate up to limit
    # If lpf[i] is 0, i is prime
    for i in range(2, limit + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, limit + 1, i):
                lpf[j] = i
    
    best_n = 10000
    min_val = float('inf')
    
    # Search for n in [10000, 1000000]
    for n in range(10000, max_n + 1):
        # Largest prime factor of n(n+1) is max(LPF(n), LPF(n+1))
        val = max(lpf[n], lpf[n+1])
        if val < min_val:
            min_val = val
            best_n = n
            
    return best_n
