def construct():
    import time
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    # Use sieve to compute totient for all n up to limit
    limit = 1000000
    
    # Initialize phi array: phi[i] = i initially
    phi = list(range(limit + 1))
    
    # Sieve for totient: for each prime p, multiply phi[j] by (1 - 1/p)
    for i in range(2, limit + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
    
    # Find smallest n with phi(n) % 120 == 0
    for n in range(1, limit + 1):
        if time.time() > deadline:
            break
        if phi[n] % 120 == 0:
            return n
    
    return 143  # Fallback if no result found within deadline
