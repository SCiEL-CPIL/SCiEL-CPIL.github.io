import time

def construct():
    start_time = time.time()
    timeout = 50.0  # 50s limit to be safe against 60s/90s kills
    
    N_MAX = 1000001
    lpf = [0] * (N_MAX + 1)
    
    # Sieve for Largest Prime Factor
    # lpf[i] will hold the largest prime factor of i
    for i in range(2, N_MAX + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N_MAX + 1, i):
                lpf[j] = i
                
    best_n = 10000
    best_val = float('inf')
    
    # Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    for n in range(10000, 1000001):
        # Check time
        if time.time() - start_time > timeout:
            break
            
        current_val = max(lpf[n], lpf[n+1])
        if current_val < best_val:
            best_val = current_val
            best_n = n
            
    return best_n
