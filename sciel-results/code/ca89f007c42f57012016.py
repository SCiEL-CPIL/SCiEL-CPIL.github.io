import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    MAX_N = 1000000
    MAX_V = MAX_N + 1

    # List to store largest prime factor
    # lpf[i] = 0 initially. If lpf[i] == 0 when i is reached in sieve, i is prime.
    lpf = [0] * (MAX_V + 1)
    
    # Sieve for Largest Prime Factor
    for i in range(2, MAX_V + 1):
        # Deadline check
        if time.time() > deadline:
            break
        
        if lpf[i] == 0:
            # i is prime
            # Update all multiples of i with i as a factor.
            # Since i increases, the last prime to touch j is the largest prime factor of j.
            for j in range(i, MAX_V + 1, i):
                lpf[j] = i
                
    # Search for n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    best_n = -1
    min_max_lpf = float('inf')
    
    for n in range(10000, MAX_N + 1):
        # Deadline check
        if time.time() > deadline:
            break
            
        # Calculate the metric for this n
        # Since n and n+1 are coprime, their prime factors are disjoint,
        # but we just take the max of their individual largest prime factors.
        lp_n = lpf[n]
        lp_np1 = lpf[n+1]
        
        current_max = lp_n if lp_n > lp_np1 else lp_np1
        
        if current_max < min_max_lpf:
            min_max_lpf = current_max
            best_n = n
            
    # If loop was broken by deadline, best_n might be -1 if we didn't find anything?
    # But 10000 iterations are fast.
    if best_n == -1:
        # Fallback: return a valid number, e.g., 10000
        return 10000
        
    return best_n
