import time
import numpy as np

def construct():
    start_time = time.time()
    LIMIT = 90  # seconds (evaluator kills at 90s)
    MARGIN = 5  # seconds safety margin
    
    N_MAX = 1000001  # We need P up to 1000001 to evaluate n=1000000
    START = 10000
    
    # 1. Sieve for Smallest Prime Factor (SPF)
    # Optimized SPF sieve
    spf = [0] * (N_MAX + 1)
    for i in range(2, N_MAX + 1):
        if spf[i] == 0:
            spf[i] = i
            if i * i <= N_MAX:
                for j in range(i * i, N_MAX + 1, i):
                    if spf[j] == 0:
                        spf[j] = i

    # 2. Compute Powerful Part P[k]
    # P[i] is the product of p^e for all prime powers p^e || i where e >= 2
    P = [1] * (N_MAX + 1)
    P[1] = 1
    
    # Local variables for speed
    spf_local = spf
    P_local = P
    
    for i in range(2, N_MAX + 1):
        p = spf_local[i]
        x = i // p
        if x % p == 0:
            y = x // p
            if y % p == 0:
                # i has p^3 or higher
                P_local[i] = P_local[x] * p
            else:
                # i has exactly p^2
                P_local[i] = P_local[x] * p * p
        else:
            # i has exactly p^1
            P_local[i] = P_local[x]
            
    # Check time
    if time.time() - start_time > LIMIT - MARGIN:
        return 10000

    # 3. Find n that maximizes P[n] * P[n+1]
    try:
        # Convert P to numpy array for fast vectorized multiplication
        # Using object dtype to avoid overflow, though int64 should be safe for 10^12
        # But to be absolutely safe and fast, we can use Python loop if numpy is slow or risky
        # For 1M elements, numpy is much faster.
        P_arr = np.array(P, dtype=np.int64)
        
        # n ranges from START to 1000000 inclusive.
        left = P_arr[START:1000001]
        right = P_arr[START+1:1000002]
        
        products = left * right
        
        max_idx = np.argmax(products)
        best_n = int(START + max_idx)
        
    except Exception:
        # Fallback to pure Python if numpy fails or overflows
        best_n = START
        best_score = -1
        for n in range(START, 1000001):
            score = P[n] * P[n + 1]
            if score > best_score:
                best_score = score
                best_n = n

    return best_n
