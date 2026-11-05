import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline to be safe
    
    MAX_N = 1000000
    # We need smooth parts for n+1 to n+24, so up to MAX_N + 24
    # But n is up to MAX_N, so we need up to MAX_N + 24
    # Let's allocate MAX_N + 25
    smooth = np.ones(MAX_N + 25, dtype=np.int64)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Compute 23-smooth parts
    for p in primes:
        power = p
        while power <= MAX_N + 24:
            smooth[power::power] *= p
            power *= p
    
    best_n = 1000
    best_count = 25  # Max possible is 24, but 25 is safe initial
    
    # We can search in chunks to check time
    # Or just loop
    for n in range(1000, MAX_N + 1):
        if time.time() > deadline:
            break
        # Extract smooth parts for n+1 to n+24
        # Use a set to count distinct values
        # To speed up, we can use a list and sort, or just a set
        # A set of 24 integers is very fast
        parts = smooth[n+1:n+25].tolist()
        count = len(set(parts))
        if count < best_count:
            best_count = count
            best_n = n
            # If we reach 1, we can stop
            if best_count == 1:
                break
                
    return best_n
