import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    limit = 10000

    # Sieve of Eratosthenes
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, math.isqrt(limit) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = bytearray(len(is_prime[i * i::i]))

    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n = len(primes)

    if n < 2:
        return primes

    # parent[i][j] = k such that L(k, i) was used to compute L(i, j)
    # We use a list of lists for parents.
    parent = [[-1] * n for _ in range(n)]
    
    # suffix_max_val[k] is a list of length k
    # suffix_max_val[k][m] = max L(j, k) for j in [m, k-1]
    # suffix_max_idx[k][m] = argmax j
    suffix_max_val = [None] * n
    suffix_max_idx = [None] * n

    best_len = 2
    best_i = 0
    best_j = 1

    for i in range(n):
        if time.time() > deadline:
            break
            
        if i == 0:
            continue

        # col_vals[k] for k in 0..i-1
        col_vals = [0] * i
        col_idxs = [-1] * i # Not strictly needed if we store parent directly, but good for building suffix max

        m_ptr = 0
        
        for k in range(i):
            # Calculate threshold
            # Condition: p[k] - p[m] <= p[i] - p[k] => p[m] >= 2*p[k] - p[i]
            threshold = 2 * primes[k] - primes[i]
            
            # Advance m_ptr to the first index where primes[m_ptr] >= threshold
            # Note: m must be < k
            while m_ptr < k and primes[m_ptr] < threshold:
                m_ptr += 1
                
            if m_ptr < k:
                # Valid m exists in [m_ptr, k-1]
                # Get max L(m, k) for m in [m_ptr, k-1]
                # This is stored in suffix_max_val[k][m_ptr]
                val = suffix_max_val[k][m_ptr]
                idx = suffix_max_idx[k][m_ptr]
                
                col_vals[k] = 1 + val
                parent[k][i] = idx
            else:
                # No valid m < k
                col_vals[k] = 2
                parent[k][i] = -1

            if col_vals[k] > best_len:
                best_len = col_vals[k]
                best_i = k
                best_j = i

        # Build suffix max for column i
        # We need suffix_max_val[i] and suffix_max_idx[i]
        # suffix_max_val[i][m] = max(col_vals[i][m], suffix_max_val[i][m+1])
        # Base case: m = i-1
        # suffix_max_val[i][i-1] = col_vals[i][i-1]
        
        sm_val = [0] * i
        sm_idx = [0] * i
        
        if i > 0:
            sm_val[i-1] = col_vals[i-1]
            sm_idx[i-1] = i-1 # The index m itself is i-1
            
            for m in range(i-2, -1, -1):
                if col_vals[m] >= sm_val[m+1]:
                    sm_val[m] = col_vals[m]
                    sm_idx[m] = m
                else:
                    sm_val[m] = sm_val[m+1]
                    sm_idx[m] = sm_idx[m+1]
                    
        suffix_max_val[i] = sm_val
        suffix_max_idx[i] = sm_idx

    # Reconstruct
    if best_len <= 2:
        # Fallback
        values = [primes[0]]
        for p in primes[1:]:
            if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                values.append(p)
        return values

    seq = [primes[best_j], primes[best_i]]
    i = best_i
    j = best_j
    while parent[i][j] != -1:
        k = parent[i][j]
        seq.append(primes[k])
        j = i
        i = k

    seq.reverse()
    return seq
