import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False

    primes = [i for i, v in enumerate(is_prime) if v]
    N = len(primes)

    if N == 0:
        return []
    if N == 1:
        return primes

    # L[i][j] is length of longest seq ending with (primes[i], primes[j])
    # parent[i][j] is h such that L[h][i] was used to compute L[i][j]
    L = [[0] * N for _ in range(N)]
    parent = [[-1] * N for _ in range(N)]
    
    # Initialize base cases: sequences of length 2
    for h in range(N):
        for i in range(h + 1, N):
            L[h][i] = 2

    best_len = 2
    best_i = 0
    best_j = 1

    for i in range(1, N - 1):
        if time.time() > deadline:
            break
            
        # Compute SufMax and SufIdx for L[h][i] for h in [0, i-1]
        SufMax = [0] * i
        SufIdx = [-1] * i
        
        current_max = 0
        current_idx = -1
        for h in range(i - 1, -1, -1):
            val = L[h][i]
            if val > current_max:
                current_max = val
                current_idx = h
            SufMax[h] = current_max
            SufIdx[h] = current_idx
            
        # Now compute L[i][j] for j in [i+1, N-1]
        h_ptr = i - 1
        
        for j in range(i + 1, N):
            target = 2 * primes[i] - primes[j]
            
            while h_ptr > 0 and primes[h_ptr - 1] >= target:
                h_ptr -= 1
                
            max_len_pre = SufMax[h_ptr]
            if max_len_pre > 0:
                new_len = max_len_pre + 1
                L[i][j] = new_len
                parent[i][j] = SufIdx[h_ptr]
                
                if new_len > best_len:
                    best_len = new_len
                    best_i = i
                    best_j = j
            # else L[i][j] remains 2 (base case)

    # Reconstruct
    path = []
    curr_i = best_i
    curr_j = best_j
    
    path.append(primes[curr_j])
    path.append(primes[curr_i])
    
    while parent[curr_i][curr_j] != -1:
        h = parent[curr_i][curr_j]
        path.append(primes[h])
        curr_j = curr_i
        curr_i = h
        
    path.reverse()
    return path
