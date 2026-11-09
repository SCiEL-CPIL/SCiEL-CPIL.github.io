def construct():
    import math
    
    MAX_N = 1000000
    MIN_N = 10000
    LIMIT = MAX_N + 1
    
    # Sieve for Largest Prime Factor
    lpf = [0] * (LIMIT + 1)
    
    for p in range(2, LIMIT + 1):
        if lpf[p] == 0:
            # p is prime
            for j in range(p, LIMIT + 1, p):
                lpf[j] = p
                
    best_n = -1
    best_score = float('inf')
    
    for n in range(MIN_N, MAX_N + 1):
        score = max(lpf[n], lpf[n+1])
        if score < best_score:
            best_score = score
            best_n = n
            
    return best_n
