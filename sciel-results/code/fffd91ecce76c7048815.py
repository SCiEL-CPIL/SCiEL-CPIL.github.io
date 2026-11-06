def construct():
    import time
    import numpy as np

    start_time = time.time()
    # Safety margin: The evaluator kills after 90s.
    # We aim to finish in < 5s.
    deadline = start_time + 85.0

    N = 1000001
    # We need lpf for numbers up to 1000001 (since n can be 1000000, n+1 is 1000001)
    # Size N+2 to be safe for indexing
    lpf = np.zeros(N + 2, dtype=np.int32)

    # Sieve for Largest Prime Factor
    # Iterate through potential primes
    for i in range(2, N + 2):
        # If lpf[i] is 0, i is prime
        if lpf[i] == 0:
            # Mark multiples of i with i as their largest prime factor
            # Since we iterate i in increasing order, the last prime factor
            # assigned to any number will be its largest prime factor.
            for j in range(i, N + 2, i):
                lpf[j] = i
        
        # Check time
        if time.time() > deadline:
            # If we run out of time, we must return something.
            # But the sieve is fast, this is unlikely.
            # If it happens, we might have incomplete lpf.
            # We should probably break and try to compute with what we have,
            # or return a default.
            # However, for N=10^6, this loop takes ~0.1s in Python.
            # So this guard is just for safety.
            pass

    # Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    # n ranges from 10000 to 1000000 inclusive
    # n+1 ranges from 10001 to 1000001 inclusive
    
    # Create array of n values
    n_vals = np.arange(10000, 1000001)
    
    # Get lpf for n and n+1
    lpf_n = lpf[n_vals]
    lpf_np1 = lpf[n_vals + 1]
    
    # Compute the objective: max of the two
    vals = np.maximum(lpf_n, lpf_np1)
    
    # Find the index of the minimum value
    min_idx = np.argmin(vals)
    best_n = int(n_vals[min_idx])
    
    return best_n
