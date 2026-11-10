import math

def construct():
    N = 123457
    # Sieve for Largest Prime Factor
    lpf = [0] * (N + 1)
    lpf[1] = 1
    
    for i in range(2, N + 1):
        if lpf[i] == 0: # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    best_val = float('inf')
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        b = N - a
        val = max(lpf[a], lpf[b])
        if val < best_val:
            best_val = val
            best_a = a
            best_b = b
            
    return [best_a, best_b]
