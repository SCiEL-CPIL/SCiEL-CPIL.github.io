import math
import time
import bisect
from array import array

def construct():
    start_time = time.time()
    deadline = start_time + 80.0 # 10s margin
    
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, p in enumerate(is_prime) if p]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return primes

    # dp[i][j] and pred[i][j]
    # Using lists of arrays for better memory layout?
    # Or just 2D lists. N=1229, N^2=1.5M.
    # 1.5M ints in Python list ~ 40MB.
    # array('i') ~ 6MB.
    
    dp = [array('i', [0]) * N for _ in range(N)]
    pred = [array('i', [-1]) * N for _ in range(N)]
    
    # suf_max[i] is array of length i
    # suf_arg[i] is array of length i
    suf_max = [None] * N
    suf_arg = [None] * N
    
    for j in range(1, N):
        if time.time() > deadline:
            break
            
        for i in range(j):
            gap = primes[j] - primes[i]
            target = primes[i] - gap
            k_min = bisect.bisect_left(primes, target, 0, i)
            
            if k_min < i:
                best_len = suf_max[i][k_min]
                best_k = suf_arg[i][k_min]
                dp[i][j] = best_len + 1
                pred[i][j] = best_k
            else:
                dp[i][j] = 2
                pred[i][j] = -1
        
        # Build suffix max and argmax for column j
        # We need suf_max[j] and suf_arg[j]
        # They will be used when j becomes 'i'
        sm = array('i', [0]) * j
        sa = array('i', [-1]) * j
        cur_max = 0
        cur_arg = -1
        for k in range(j - 1, -1, -1):
            val = dp[k][j]
            if val > cur_max:
                cur_max = val
                cur_arg = k
            sm[k] = cur_max
            sa[k] = cur_arg
        suf_max[j] = sm
        suf_arg[j] = sa
        
    # Find max
    max_len = 0
    end_i = 0
    end_j = 1
    for j in range(1, N):
        if time.time() > deadline:
            break
        for i in range(j):
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                end_i = i
                end_j = j
                
    # Reconstruct
    path = []
    i, j = end_i, end_j
    while j > 0:
        path.append(primes[j])
        k = pred[i][j]
        if k == -1:
            path.append(primes[i])
            break
        i, j = k, i
        
    path.reverse()
    return path
