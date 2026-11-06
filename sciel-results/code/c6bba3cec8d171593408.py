def construct():
    import time
    start = time.time()
    limit = 1000000
    # Sieve for phi
    phi = list(range(limit + 1))
    for i in range(2, limit + 1):
        if phi[i] == i:
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
    
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
            
    return 1 # Fallback
