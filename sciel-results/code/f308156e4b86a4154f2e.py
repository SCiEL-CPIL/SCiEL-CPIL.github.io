import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    N = 1000000
    limit = N + 1
    
    lpf = [0] * (limit + 1)
    
    # Sieve for Largest Prime Factor
    for i in range(2, limit + 1):
        if time.time() > deadline:
            break
        if lpf[i] == 0:
            for j in range(i, limit + 1, i):
                lpf[j] = i
                
    best_val = float('inf')
    best_n = 10000
    
    # Scan for minimum max prime factor
    for n in range(10000, N + 1):
        if time.time() > deadline:
            break
        val = lpf[n]
        if lpf[n+1] > val:
            val = lpf[n+1]
        if val < best_val:
            best_val = val
            best_n = n
            
    return best_n
