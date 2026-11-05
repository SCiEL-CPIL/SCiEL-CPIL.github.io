import time

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin
    
    MAX_N = 1000000
    
    # 1. Sieve for Largest Prime Factor (LPF)
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (MAX_N + 2)  # +2 to safely access n+1 when n=MAX_N
    
    # We need primes up to MAX_N to sieve correctly?
    # Actually, we can just iterate i from 2 to MAX_N.
    # If i is prime, it is a prime factor for its multiples.
    # Standard sieve approach:
    # for i in range(2, MAX_N + 1):
    #     if lpf[i] == 0: # i is prime
    #         for j in range(i, MAX_N + 2, i):
    #             lpf[j] = i
    
    # This is O(N log log N).
    # N = 1,000,001.
    # This will run in < 1 second in Python.
    
    for i in range(2, MAX_N + 2):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, MAX_N + 2, i):
                lpf[j] = i
                
    # 2. Scan for the best n
    best_n = 10000
    best_val = float('inf')
    
    # We can scan the entire range now
    # Check time occasionally if it's extremely slow, but it shouldn't be.
    
    for n in range(10000, MAX_N + 1):
        if time.time() > deadline:
            break
            
        val_n = lpf[n]
        val_np1 = lpf[n + 1]
        
        current_val = val_n if val_n > val_np1 else val_np1
        
        if current_val < best_val:
            best_val = current_val
            best_n = n
            
    return best_n

# EVOLVE-BLOCK-END
