import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin out of 90s
    
    N_lim = 1000001  # We need lpf up to 1000001 (for n+1 when n=1000000)
    
    # Optimized sieve: lpf[i] stores the largest prime factor of i
    lpf = [0] * (N_lim + 1)
    
    for i in range(2, N_lim + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N_lim + 1, i):
                lpf[j] = i
                
    best_n = 10000
    best_m = float('inf')
    
    # Full scan with time guard
    # We scan all n in [10000, 1000000]
    for n in range(10000, 1000001):
        m1 = lpf[n]
        m2 = lpf[n+1]
        m = m1 if m1 > m2 else m2
        
        if m < best_m:
            best_m = m
            best_n = n
        
        # Check time every 50000 iterations to reduce overhead while staying safe
        if (n - 10000) % 50000 == 0 and time.time() > deadline:
            break
            
    return best_n
