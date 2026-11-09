import time
import math
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    limit = 10000

    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, math.isqrt(limit) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = bytearray(len(is_prime[i * i::i]))

    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n = len(primes)

    if n < 2:
        return primes

    dp = [None] * n
    parent = [None] * n

    for j in range(n):
        dp[j] = [2] * j
        parent[j] = [-1] * j

    best_len = 2
    best_i = 0
    best_j = 1

    for i in range(1, n - 1):
        if time.time() > deadline:
            break

        dpi = dp[i]
        suffix_max = [0] * i
        suffix_argmax = [-1] * i
        
        suffix_max[i-1] = dpi[i-1]
        suffix_argmax[i-1] = i-1
        
        for k in range(i-2, -1, -1):
            if dpi[k] > suffix_max[k+1]:
                suffix_max[k] = dpi[k]
                suffix_argmax[k] = k
            else:
                suffix_max[k] = suffix_max[k+1]
                suffix_argmax[k] = suffix_argmax[k+1]

        for j in range(i + 1, n):
            if time.time() > deadline:
                break

            threshold = 2 * primes[i] - primes[j]
            
            # Find leftmost k in [0, i) with primes[k] >= threshold
            pos = bisect.bisect_left(primes, threshold, 0, i)
            
            if pos < i:
                new_len = suffix_max[pos] + 1
                if new_len > dp[j][i]:
                    dp[j][i] = new_len
                    parent[j][i] = suffix_argmax[pos]
                    if new_len > best_len:
                        best_len = new_len
                        best_i = i
                        best_j = j

    if time.time() > deadline:
        values = [primes[0]]
        for p in primes[1:]:
            if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                values.append(p)
        return values

    if best_len <= 2:
        values = [primes[0]]
        for p in primes[1:]:
            if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                values.append(p)
        return values

    seq = [primes[best_j], primes[best_i]]
    i_cur = best_i
    j_cur = best_j
    
    while parent[j_cur][i_cur] != -1:
        k = parent[j_cur][i_cur]
        seq.append(primes[k])
        j_cur = i_cur
        i_cur = k
        
    seq.reverse()
    return seq
