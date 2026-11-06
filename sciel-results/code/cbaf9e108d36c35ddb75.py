import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    TARGET = 123457
    
    # Compute largest prime factor for all numbers up to TARGET
    # lp[i] = largest prime factor of i
    lp = [0] * (TARGET + 1)
    lp[1] = 1
    
    # Sieve to compute largest prime factor
    for i in range(2, TARGET + 1):
        if lp[i] == 0:  # i is prime
            for j in range(i, TARGET + 1, i):
                lp[j] = i  # i is a prime factor of j; since we go in increasing order, last write wins
    
    # Now lp[n] contains the largest prime factor of n
    # Scan all pairs (a, b) where a + b = TARGET, a >= 1, b >= 1
    best_max = TARGET + 1
    best_pair = (1, TARGET - 1)
    
    for a in range(1, TARGET):
        b = TARGET - a
        max_p = max(lp[a], lp[b])
        if max_p < best_max:
            best_max = max_p
            best_pair = (a, b)
            # Early termination if we found the theoretical minimum
            if best_max == 2:
                break
    
    return [best_pair[0], best_pair[1]]
