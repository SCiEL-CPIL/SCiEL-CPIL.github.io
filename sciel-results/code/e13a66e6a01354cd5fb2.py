import time
import math

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s margin from 90s limit
    
    # 1. Generate primes up to 10000
    limit_val = 10000
    is_prime = bytearray(b'\x01') * (limit_val + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit_val)) + 1):
        if is_prime[i]:
            # Slice assignment is faster
            is_prime[i*i : limit_val + 1 : i] = bytearray(len(is_prime[i*i : limit_val + 1 : i]))
    
    primes = [i for i, flag in enumerate(is_prime) if flag]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return primes
    
    # 2. DP Setup
    # dp[e][p] = length of longest valid subsequence ending at primes[e] with previous prime primes[p]
    # choice[e][p] = index k such that dp[p][k] was the max used to compute dp[e][p]
    # If no such k, choice is -1
    
    # Using list of lists for dp and choice
    dp = [[0] * N for _ in range(N)]
    choice = [[-1] * N for _ in range(N)]
    
    # For each p, we maintain a pointer to the next k to consider
    # and the current max value for the range [k_ptr+1, p-1]
    # k_ptr[p] starts at p-1. As e increases, primes[e] increases, 
    # so gap(e,p) increases, so limit = primes[p] - gap(e,p) decreases.
    # We need primes[k] >= limit. Since limit decreases, more k's (smaller indices) become valid.
    # So k_ptr moves from p-1 downwards to 0.
    k_ptr = [p - 1 for p in range(N)]
    max_val = [0] * N
    argmax_idx = [-1] * N
    
    best_len = 2
    best_e = 1
    best_p = 0
    
    # Precompute primes list for faster access
    p_list = primes
    
    # 3. DP Iteration
    for e in range(N):
        if time.time() - start_time > deadline:
            break
        pe = p_list[e]
        dp_e = dp[e]
        choice_e = choice[e]
        
        for p in range(e):
            pp = p_list[p]
            gap = pe - pp
            limit = pp - gap
            
            # Update max_val[p] with new k's that become valid
            # k must satisfy primes[k] >= limit
            # We add k's in descending order of index (which is descending order of value)
            # k_ptr starts at p-1. We decrement it.
            kp = k_ptr[p]
            mv = max_val[p]
            am = argmax_idx[p]
            dp_p = dp[p]
            
            while kp >= 0 and p_list[kp] >= limit:
                k_val = dp_p[kp]
                if k_val > mv:
                    mv = k_val
                    am = kp
                kp -= 1
            
            k_ptr[p] = kp
            max_val[p] = mv
            argmax_idx[p] = am
            
            if mv > 0:
                val = mv + 1
                dp_e[p] = val
                choice_e[p] = am
            else:
                val = 2
                dp_e[p] = val
                choice_e[p] = -1
            
            if val > best_len:
                best_len = val
                best_e = e
                best_p = p
                
    # 4. Reconstruct Sequence
    seq = []
    if best_len >= 2:
        seq.append(p_list[best_e])
        seq.append(p_list[best_p])
        curr_e = best_e
        curr_p = best_p
        k = choice[curr_e][curr_p]
        while k != -1:
            seq.append(p_list[k])
            curr_e, curr_p = curr_p, k
            k = choice[curr_e][curr_p]
        
    seq.reverse()
    return seq
