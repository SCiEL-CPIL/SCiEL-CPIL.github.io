import time

def construct():
    start_time = time.time()
    limit = 1000000
    # We need LPF for n and n+1, so up to 1000001
    max_val = limit + 1
    
    # Sieve to find largest prime factor
    # Using a list of ints. lpf[i] will store the largest prime factor of i.
    lpf = [0] * (max_val + 1)
    
    # Standard sieve logic modified to store largest prime factor
    # Iterate p from 2 to max_val.
    # If lpf[p] is 0, p is prime.
    # Update lpf[j] for all multiples j of p.
    # Since p increases, the final assignment is by the largest prime factor.
    
    for p in range(2, max_val + 1):
        if lpf[p] == 0:
            # p is prime
            # Update multiples
            # Optimization: use local variable for lpf
            lpf_local = lpf
            for j in range(p, max_val + 1, p):
                lpf_local[j] = p
                
    # Check time, though this should be fast (< 1s)
    elapsed = time.time() - start_time
    if elapsed > 50:
        # Fallback if sieve is too slow (highly unlikely)
        return 10000

    best_n = 10000
    min_score = float('inf')
    
    # Scan for the minimum max(LPF(n), LPF(n+1))
    lpf_local = lpf
    for n in range(10000, limit + 1):
        # Score is the max of the largest prime factors of n and n+1
        score = max(lpf_local[n], lpf_local[n+1])
        if score < min_score:
            min_score = score
            best_n = n
            
        # Deadline guard: if we are running out of time, return best found so far
        # The scan is very fast, so this is unlikely to trigger, but good practice.
        if time.time() - start_time > 55:
            break

    return best_n
