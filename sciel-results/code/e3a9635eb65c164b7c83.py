import time
import math
import numpy as np

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

    # dp_val[i] will be a list of length i, where dp_val[i][k] = L(k, i)
    # dp_parent[i][j] = k such that L(k, i) was used to compute L(i, j)
    # We need a 2D array for parents.
    # parent[i][j] is valid for i < j.
    parent = [[-1] * n for _ in range(n)]
    
    # st[i] will be the Sparse Table for column i
    # st[i] is a list of arrays, st[i][k] is the max value for range of length 2^k
    # st_idx[i] is the corresponding index
    # We build these on the fly.
    
    # To save memory and time, we can just store the values in a list of lists
    # and build the ST only when needed?
    # No, we build ST for column i after processing all j > i?
    # No, we build ST for column i after processing all k < i (i.e., after inner loop for i).
    # Then we use it for j > i.
    
    col_vals = [None] * n  # col_vals[i] = list of L(k, i) for k < i
    st_val = [None] * n
    st_idx = [None] * n

    best_len = 2
    best_i = 0
    best_j = 1

    # Precompute logs for sparse table
    # max log2(n)
    max_log = int(math.log2(n)) + 1

    for i in range(n):
        if time.time() > deadline:
            break
            
        # Initialize col_vals[i]
        if i == 0:
            col_vals[i] = []
            continue
            
        # Compute L(k, i) for k in 0..i-1
        vals_i = [0] * i
        for k in range(i):
            # Compute L(k, i)
            # Need max L(m, k) for m in [k_min, k-1]
            # Condition: primes[k] - primes[m] <= primes[i] - primes[k]
            # primes[m] >= 2*primes[k] - primes[i]
            threshold = 2 * primes[k] - primes[i]
            
            # Find k_min (actually m_min) such that primes[m] >= threshold
            # m is in [0, k-1]
            # Since primes is sorted, we can binary search in primes[0:k]
            # But we can just use bisect
            import bisect
            m_min = bisect.bisect_left(primes, threshold, 0, k)
            
            if m_min < k:
                # Query st_val[k] for range [m_min, k-1]
                if st_val[k] is not None:
                    # Get max and argmax
                    max_val, argmax_m = _query_st(st_val[k], st_idx[k], m_min, k - 1)
                    if max_val > 0:
                        vals_i[k] = 1 + max_val
                        parent[k][i] = argmax_m
                    else:
                        vals_i[k] = 2
                        parent[k][i] = -1
                else:
                    # This shouldn't happen if k < i and we built ST for k previously
                    vals_i[k] = 2
                    parent[k][i] = -1
            else:
                vals_i[k] = 2
                parent[k][i] = -1
                
            if vals_i[k] > best_len:
                best_len = vals_i[k]
                best_i = k
                best_j = i

        col_vals[i] = vals_i
        
        # Build Sparse Table for column i
        if i > 0:
            st_val[i], st_idx[i] = _build_st(vals_i)

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

def _build_st(vals):
    n = len(vals)
    if n == 0:
        return None, None
    max_log = int(math.log2(n)) + 1
    st_val = [vals[:]]
    st_idx = [list(range(n))]
    
    for k in range(1, max_log):
        if (1 << k) > n:
            break
        prev_val = st_val[k-1]
        prev_idx = st_idx[k-1]
        length = n - (1 << k) + 1
        curr_val = [0] * length
        curr_idx = [0] * length
        half = 1 << (k-1)
        for i in range(length):
            v1 = prev_val[i]
            v2 = prev_val[i + half]
            if v1 >= v2:
                curr_val[i] = v1
                curr_idx[i] = prev_idx[i]
            else:
                curr_val[i] = v2
                curr_idx[i] = prev_idx[i + half]
        st_val.append(curr_val)
        st_idx.append(curr_idx)
        
    return st_val, st_idx

def _query_st(st_val, st_idx, l, r):
    if l > r:
        return 0, -1
    length = r - l + 1
    k = int(math.log2(length))
    v1 = st_val[k][l]
    v2 = st_val[k][r - (1 << k) + 1]
    if v1 >= v2:
        return v1, st_idx[k][l]
    else:
        return v2, st_idx[k][r - (1 << k) + 1]
