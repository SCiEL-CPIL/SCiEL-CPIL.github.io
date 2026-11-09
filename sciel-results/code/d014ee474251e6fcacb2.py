import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit out of 60s to be safe
    
    MAX_N = 1000000
    LIMIT = MAX_N + 1  # We need lpf up to 1000001
    
    # Sieve for Largest Prime Factor
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (LIMIT + 1)
    
    # Iterate through potential primes
    for i in range(2, LIMIT + 1):
        if lpf[i] == 0:
            # i is prime
            # Mark all multiples of i with i as their largest prime factor
            # Since we iterate i in increasing order, the last update
            # for any number j will be by its largest prime factor.
            for j in range(i, LIMIT + 1, i):
                lpf[j] = i
                if time.time() > deadline:
                    # Fallback to seed if sieve takes too long (unlikely)
                    return 10000

    best_n = 10000
    best_val = float('inf')
    
    # Scan for the optimal n
    for n in range(10000, MAX_N + 1):
        if time.time() > deadline:
            break
        
        lpf_n = lpf[n]
        lpf_np1 = lpf[n + 1]
        current_val = lpf_n if lpf_n > lpf_np1 else lpf_np1
        
        if current_val < best_val:
            best_val = current_val
            best_n = n

    return best_n
