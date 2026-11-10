import time
import numpy as np
from collections import defaultdict

MAX_VAL = 10_000_000
TIME_LIMIT = 85  # seconds, safety margin before 90s kill

def _sigma_linear_sieve(n):
    """Compute sigma(k) for all k in [1, n] using a linear sieve in O(n)."""
    sigma = np.zeros(n + 1, dtype=np.int64)
    spf = np.zeros(n + 1, dtype=np.int64)  # smallest prime factor
    cnt = np.zeros(n + 1, dtype=np.int64)  # exponent of spf in n
    p_pow = np.zeros(n + 1, dtype=np.int64)  # spf^n's exponent power
    sigma[1] = 1
    spf[1] = 1
    p_pow[1] = 1
    
    primes = []
    for i in range(2, n + 1):
        if spf[i] == 0:
            spf[i] = i
            cnt[i] = 1
            p_pow[i] = i
            sigma[i] = 1 + i
            primes.append(i)
        
        for p in primes:
            if p > spf[i] or i * p > n:
                break
            np_ = i * p
            spf[np_] = p
            if p == spf[i]:
                cnt[np_] = cnt[i] + 1
                p_pow[np_] = p_pow[i] * p
                # sigma[i*p] = sigma[i] + p^(k+1) * sigma[m]
                # where i = p^k * m, sigma[m] = sigma[i] / (1+p+...+p^k)
                k = cnt[i]
                # (p^(k+1) - 1) // (p - 1) is the sum 1+p+...+p^k
                pk1_sum = (p_pow[i] * p - 1) // (p - 1)
                m_sigma = sigma[i] // pk1_sum
                pk2 = p_pow[i] * p * p
                pk2_sum = (pk2 - 1) // (p - 1)
                sigma[np_] = m_sigma * pk2_sum
            else:
                cnt[np_] = 1
                p_pow[np_] = p
                sigma[np_] = sigma[i] * (1 + p)
    return sigma

def construct():
    start_time = time.time()
    
    # Compute sigma for all n up to MAX_VAL using linear sieve
    sigma_arr = _sigma_linear_sieve(MAX_VAL)
    
    # Phase 1: Targeted search near 3/2 ratio
    # For each b, check if a = round(3b/2) or nearby values have matching sigma
    best_num = None
    best_den = None
    best_pair = (11, 6)  # fallback: sigma(11)=12, sigma(6)=12
    
    # Check near 3/2: for each b, look at a in {floor(3b/2), ceil(3b/2), and a few neighbors}
    # This is O(n * k) where k is small
    for b in range(2, MAX_VAL + 1):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        target = 3 * b // 2
        # Check a values around 3b/2
        for da in range(-3, 4):
            a = target + da
            if a < 1 or a > MAX_VAL:
                continue
            if a == b:
                continue
            if 2 * a == 3 * b:
                continue  # exact 3/2, invalid
            if sigma_arr[a] == sigma_arr[b]:
                error_num = abs(2 * a - 3 * b)
                error_den = 2 * b
                if best_num is None or error_num * best_den < best_num * error_den:
                    best_num = error_num
                    best_den = error_den
                    best_pair = (a, b)
                if error_num <= 1:
                    break
    
    if best_num is not None and best_num <= 1:
        a, b = best_pair
        return [int(a), int(b)]
    
    # Phase 2: Fiber-based search for remaining time
    # Build fibers but only for sigma values that appear frequently
    # Use a hash-based approach to group
    if time.time() - start_time > TIME_LIMIT - 10:
        a, b = best_pair
        return [int(a), int(b)]
    
    # Build fiber map - group by sigma value
    # To save memory, use a dict of lists
    fiber_map = defaultdict(list)
    for n in range(1, MAX_VAL + 1):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        s = int(sigma_arr[n])
        fiber_map[s].append(n)
    
    # Process fibers, prioritizing larger ones
    fibers_sorted = sorted(fiber_map.items(), key=lambda x: len(x[1]), reverse=True)
    
    for s_val, vals in fibers_sorted:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        n_vals = len(vals)
        if n_vals < 2:
            continue
        
        # Two-pointer: for each b, find a closest to 3b/2
        a_idx = 0
        for b_idx in range(n_vals):
            b = vals[b_idx]
            target = 3 * b / 2.0
            
            while a_idx < n_vals - 1 and vals[a_idx] < target:
                a_idx += 1
            
            candidates_a = set()
            if a_idx < n_vals:
                candidates_a.add(vals[a_idx])
            if a_idx > 0:
                candidates_a.add(vals[a_idx - 1])
            
            for a in candidates_a:
                if a == b:
                    continue
                if 2 * a == 3 * b:
                    continue
                error_num = abs(2 * a - 3 * b)
                error_den = 2 * b
                
                if best_num is None or error_num * best_den < best_num * error_den:
                    best_num = error_num
                    best_den = error_den
                    best_pair = (a, b)
        
        # Early termination if error is very small
        if best_num is not None and best_num <= 1:
            break
    
    a, b = best_pair
    return [int(a), int(b)]
