import numpy as np
import time

def construct():
    deadline = time.time() + 80
    MAX = 1000024
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    sp = np.arange(MAX + 1, dtype=np.int64)
    for p in PRIMES:
        for m in range(p, MAX + 1, p):
            while sp[m] % p == 0:
                sp[m] //= p
    
    # Compress smooth values to small integers for efficient counting
    unique_vals, inv = np.unique(sp[1001:1000025], return_inverse=True)
    num_unique = len(unique_vals)
    
    # inv[i] corresponds to sp[1001+i], i.e., index 1001+i
    # We want windows of size 24 starting at n=1000: elements sp[1001]..sp[1024]
    # inv indices 0..23 correspond to sp[1001]..sp[1024]
    
    # Use a sliding window with a count array for compressed values
    W = 24
    N = len(inv)  # number of elements we track
    best_n = 1000
    best_cnt = W + 1
    
    # Count array for compressed values
    cnt = np.zeros(num_unique, dtype=np.int32)
    distinct = 0
    
    # Initialize window for n=1000: inv[0..23]
    for i in range(W):
        if cnt[inv[i]] == 0:
            distinct += 1
        cnt[inv[i]] += 1
    
    if distinct < best_cnt:
        best_cnt = distinct
        best_n = 1000
    
    # Slide window
    for start in range(1, N - W + 1):
        # Remove inv[start-1]
        idx_out = inv[start - 1]
        cnt[idx_out] -= 1
        if cnt[idx_out] == 0:
            distinct -= 1
        # Add inv[start+W-1]
        idx_in = inv[start + W - 1]
        if cnt[idx_in] == 0:
            distinct += 1
        cnt[idx_in] += 1
        
        n = 1000 + start
        if distinct < best_cnt:
            best_cnt = distinct
            best_n = n
            if best_cnt <= 1:
                break
        
        if time.time() > deadline:
            break
    
    return best_n
