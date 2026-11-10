import time

def construct():
    MAX_N = 1000000
    # We need lpf for indices up to MAX_N + 1
    limit = MAX_N + 1
    
    # Sieve for largest prime factor
    # lpf[i] will store the largest prime factor of i
    lpf = [1] * (limit + 1)
    
    # Standard sieve modification
    # Iterate i from 2 to limit
    # If lpf[i] is 1, i is prime
    # Then for all multiples j of i, set lpf[j] = i
    # Since we iterate i in increasing order, the last prime factor assigned
    # will be the largest one.
    
    for i in range(2, limit + 1):
        if lpf[i] == 1:
            for j in range(i, limit + 1, i):
                lpf[j] = i
                
    # Find n in [10000, MAX_N] that minimizes max(lpf[n], lpf[n+1])
    best_n = -1
    best_val = float('inf')
    
    for n in range(10000, MAX_N + 1):
        # Calculate largest prime factor of n(n+1)
        # It is max(lpf[n], lpf[n+1])
        current_val = lpf[n]
        if lpf[n+1] > current_val:
            current_val = lpf[n+1]
            
        if current_val < best_val:
            best_val = current_val
            best_n = n
            
    return best_n
