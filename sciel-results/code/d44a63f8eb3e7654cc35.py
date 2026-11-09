import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    N = 1000001  # max value of n+1 when n=1000000
    
    # Sieve to compute largest prime factor for each number up to N
    lpf = [0] * (N + 1)
    
    for i in range(2, N + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    # Scan for best pair - return n, not the value
    best_n = 10000
    best_val = max(lpf[10000], lpf[10001])
    
    for n in range(10001, 1000001):
        if time.time() - start_time > DEADLINE:
            break
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
    
    return best_n
