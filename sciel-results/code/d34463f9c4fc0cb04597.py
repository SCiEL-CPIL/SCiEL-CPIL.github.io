import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin
    
    MAX_VAL = 1000001  # need n+1 up to 1000001
    
    # Build smallest prime factor (SPF) sieve
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute largest prime factor for all numbers up to MAX_VAL
    lpf_arr = [1] * (MAX_VAL + 1)
    # Optimization: Use local variables for speed
    spf_local = spf
    lpf_local = lpf_arr
    for x in range(2, MAX_VAL + 1):
        t = x
        last_prime = 1
        while t > 1:
            p = spf_local[t]
            last_prime = p
            while t % p == 0:
                t //= p
        lpf_local[x] = last_prime
    
    # Find n in [10000, 1000000] minimizing max(LPF(n), LPF(n+1))
    best_n = 10000
    best_val = max(lpf_arr[10000], lpf_arr[10001])
    
    # Optimization: Local variable access in loop
    lpf_arr_local = lpf_arr
    for n in range(10001, 1000001):
        # Time check every 10000 iterations to reduce overhead
        if (n - 10001) % 10000 == 0 and time.time() > deadline:
            break
        lpf_n = lpf_arr_local[n]
        lpf_n1 = lpf_arr_local[n + 1]
        cur = lpf_n if lpf_n > lpf_n1 else lpf_n1
        if cur < best_val:
            best_val = cur
            best_n = n
            
    return best_n
