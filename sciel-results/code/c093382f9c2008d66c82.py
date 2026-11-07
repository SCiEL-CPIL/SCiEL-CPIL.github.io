import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s hard kill
    
    # We need to find M <= 10^18 with many k such that k*sigma(k) = M
    # k <= sqrt(M)
    
    # Strategy: Compute k*sigma(k) for many k and find collisions
    # Use a larger search space by being smarter about memory
    
    N = 80_000_000  # search up to 80 million
    
    # SPF sieve using numpy
    spf = np.arange(N + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1
    limit = int(math.isqrt(N))
    for i in range(2, limit + 1):
        if spf[i] == i:
            start = i * i
            step = i
            indices = np.arange(start, N + 1, step)
            mask = spf[indices] == indices
            if mask.any():
                spf[indices[mask]] = i
    
    spf_list = spf.tolist()
    del spf
    
    # Compute sigma values
    sigma = [1] * (N + 1)
    t_check_interval = 100000
    for i in range(2, N + 1):
        if (i & (t_check_interval - 1)) == 0:
            if time.time() > deadline - 40:
                N = i
                break
        p = spf_list[i]
        m = i // p
        if m % p == 0:
            power = p
            while m % p == 0:
                m //= p
                power *= p
            sigma[i] = sigma[m] * (power * p - 1) // (p - 1)
        else:
            sigma[i] = sigma[m] * (1 + p)
    
    # Compute M = k*sigma(k) and find collisions
    # Use dict mapping M -> list of k values
    M_to_ks = {}
    best_M = 333312
    best_sols = [336, 372, 434]
    best_count = 3
    
    t_check_interval2 = 200000
    for k in range(1, N + 1):
        if (k & (t_check_interval2 - 1)) == 0:
            if time.time() > deadline - 5:
                break
        M_val = k * sigma[k]
        if M_val <= 10**18:
            if M_val in M_to_ks:
                lst = M_to_ks[M_val]
                lst.append(k)
                if len(lst) > best_count:
                    best_count = len(lst)
                    best_sols = lst[:]
                    best_M = M_val
            else:
                M_to_ks[M_val] = [k]
    
    # Verify solutions
    valid_sols = []
    isqrt_M = math.isqrt(best_M)
    for k in best_sols:
        if 1 <= k <= isqrt_M:
            # Recompute sigma(k) to verify
            n = k
            total = 1
            p = spf_list[n] if n <= N else None
            if n <= N:
                p = spf_list[n]
                m = n
                while m > 1:
                    p = spf_list[m]
                    exp = 0
                    p_sum = 1
                    p_pow = 1
                    while m % p == 0:
                        m //= p
                        exp += 1
                        p_pow *= p
                        p_sum += p_pow
                    total *= p_sum
                if k * total == best_M:
                    valid_sols.append(k)
            else:
                # For k > N, we need to verify differently
                # Just check if k <= isqrt_M
                valid_sols.append(k)
    
    if not valid_sols:
        valid_sols = [336, 372, 434]
    
    return {'M': int(best_M), 'k': [int(x) for x in valid_sols]}
