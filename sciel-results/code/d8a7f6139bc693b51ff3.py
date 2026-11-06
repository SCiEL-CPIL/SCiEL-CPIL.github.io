import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    max_val = 1000001
    lpf = [0] * (max_val + 1)
    
    # Sieve for largest prime factor
    # 2 is prime
    for m in range(2, max_val + 1, 2):
        lpf[m] = 2
        
    # Odd primes
    for p in range(3, max_val + 1, 2):
        if lpf[p] == 0:  # p is prime
            for m in range(p, max_val + 1, p):
                lpf[m] = p

    # Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = max(lpf[10000], lpf[10001])
    
    # Local variable for speed
    lpf_local = lpf
    
    for n in range(10001, 1000001):
        val = lpf_local[n]
        val_next = lpf_local[n + 1]
        if val_next > val:
            val = val_next
        if val < best_val:
            best_val = val
            best_n = n
            
        if time.time() > deadline:
            break
            
    return best_n
