import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 100.0  # Safe margin
    
    # Determine N
    # Start with a conservative N, can increase if time permits
    # 10^7 is a good balance
    N = 10000000
    
    # Check if we can afford N
    # Sieve complexity is roughly N * log(log(N))
    # 10^7 * log(log(10^7)) ~ 10^7 * 2.5 ~ 2.5 * 10^7 ops
    # NumPy vectorization helps.
    
    # Initialize phi
    phi = np.arange(N + 1, dtype=np.int64)
    
    # Sieve of Eratosthenes for phi
    # Iterate i from 2 to N
    # If phi[i] == i, i is prime
    # Update multiples of i
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            # Update phi for multiples of i
            # phi[j] -= phi[j] // i
            phi[i::i] -= phi[i::i] // i
        
        # Time check
        if time.time() > deadline - 20.0:
            # If we are running out of time, break early?
            # But phi is incomplete.
            # Better to finish the sieve or abort and use what we have?
            # If we abort, phi values are wrong.
            # We should ensure the sieve finishes.
            # If N is too big, we should have chosen smaller N.
            # Let's assume 10^7 is fast enough.
            pass

    # Scan for longest run of distinct totients
    # phi[1] = 1
    # We need last_occurrence array of size N+1
    # phi values are <= N
    last_occ = np.zeros(N + 1, dtype=np.int32)
    
    best_k = 1
    best_s = 1
    left = 1
    
    # phi[1] is 1
    last_occ[phi[1]] = 1
    
    for right in range(2, N + 1):
        val = phi[right]
        # If this value was seen at an index >= left, we have a collision
        # Move left to last_occ[val] + 1
        if last_occ[val] >= left:
            left = last_occ[val] + 1
        
        last_occ[val] = right
        
        k = right - left + 1
        if k > best_k:
            best_k = k
            best_s = left
            
        # Time check
        if time.time() > deadline - 5.0:
            break

    return {'start': best_s, 'length': best_k}
