import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55 seconds safety margin
    
    N = 1000000
    # We need lpf up to N+1 because we check n+1 for n=N
    max_val = N + 1
    
    # Sieve for Largest Prime Factor
    lpf = [0] * (max_val + 1)
    
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, max_val + 1, i):
                lpf[j] = i
                
    best_n = 10000
    best_val = float('inf')
    
    # Scan range [10000, 1000000]
    for n in range(10000, N + 1):
        # Check deadline
        if time.time() > deadline:
            break
            
        val_n = lpf[n]
        val_n1 = lpf[n+1]
        current_max = max(val_n, val_n1)
        
        if current_max < best_val:
            best_val = current_max
            best_n = n
            
    return best_n
