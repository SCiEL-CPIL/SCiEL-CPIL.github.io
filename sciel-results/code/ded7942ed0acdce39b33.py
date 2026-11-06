def construct():
    import time
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N = 1000001
    # Initialize largest prime factor array
    lpf = [0] * (N + 1)
    
    # Sieve for largest prime factor
    # i is the current candidate for prime
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                if time.time() > deadline:
                    # If we are running out of time, we might not finish the sieve.
                    # But 10^6 is very fast. This guard is just in case.
                    # We can't return a valid answer if the sieve isn't done.
                    # However, given the speed, this is unlikely.
                    pass

    # Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    min_val = float('inf')
    best_n = 10000
    
    # We can vectorize this part or just loop. 
    # Loop is fine for 10^6 iterations.
    for n in range(10000, 1000001):
        val = max(lpf[n], lpf[n+1])
        if val < min_val:
            min_val = val
            best_n = n

    return best_n
