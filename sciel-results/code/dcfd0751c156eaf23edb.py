def construct():
    N = 1000000
    lpf = [0] * (N + 1)
    
    # Sieve for Largest Prime Factor
    for i in range(2, N + 1):
        if lpf[i] == 0:
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    best_n = 10000
    best_val = max(lpf[10000], lpf[10001])
    
    for n in range(10001, N):
        val = max(lpf[n], lpf[n+1])
        if val < best_val:
            best_val = val
            best_n = n
            
    return best_n
