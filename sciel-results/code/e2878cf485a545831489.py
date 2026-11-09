import math
import bisect
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit+1, i):
                is_prime[j] = False
        
    primes = [i for i, prime in enumerate(is_prime) if prime]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return primes

    # aux[x] = (gaps, max_lens, best_ks)
    # gaps: sorted list of gaps P[x]-P[k] for k < x
    # max_lens: prefix max of lengths dp[k][x]
    # best_ks: the k that achieved the max length
    aux = [None] * N
    # parent[h][x] = k, where k is the predecessor of h in the optimal sequence ending at x
    # If k is -1, h is the start of the sequence
    parent = [[-1] * N for _ in range(N)]
    
    best_len = 1
    best_h = -1
    best_x = -1
    
    # We iterate x from 1 to N-1
    # For each x, we compute dp[h][x] for all h < x
    for x in range(1, N):
        if time.time() > deadline:
            break
        px = primes[x]
        entries = []
        
        for h in range(x):
            gap = px - primes[h]
            
            # Query aux[h]
            if aux[h] is None:
                m_len = 1
                k_star = -1
            else:
                gaps_h, max_lens_h, best_ks_h = aux[h]
                # Find rightmost index with gap <= current gap
                idx = bisect.bisect_right(gaps_h, gap) - 1
                if idx < 0:
                    m_len = 1
                    k_star = -1
                else:
                    m_len = max_lens_h[idx]
                    k_star = best_ks_h[idx]
            
            new_len = 1 + m_len
            parent[h][x] = k_star
            
            if new_len > best_len:
                best_len = new_len
                best_h = h
                best_x = x
            
            entries.append((gap, new_len, h))
        
        # Build aux[x]
        if entries:
            entries.sort(key=lambda t: t[0])
            
            gaps_x = []
            max_lens_x = []
            best_ks_x = []
            
            running_max = 0
            running_k = -1
            
            for gap, length, k_val in entries:
                if length > running_max:
                    running_max = length
                    running_k = k_val
                
                gaps_x.append(gap)
                max_lens_x.append(running_max)
                best_ks_x.append(running_k)
            
            aux[x] = (gaps_x, max_lens_x, best_ks_x)
        
        if time.time() > deadline:
            break
    
    # Reconstruct sequence
    if best_x == -1:
        return [primes[0]]
        
    seq = []
    x = best_x
    h = best_h
    
    while True:
        seq.append(primes[x])
        k = parent[h][x]
        if k == -1:
            seq.append(primes[h])
            break
        x = h
        h = k
        
    seq.reverse()
    return seq
