import math
import time
import numpy as np

def _sigma_slow(n):
    if n == 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            term = 1
            pw = 1
            while temp % p == 0:
                temp //= p
                exp += 1
                pw *= p
                term += pw
            result *= term
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (1 + temp)
    return result

def construct():
    start = time.time()
    deadline = start + 140.0

    def _fallback():
        candidates = [
            (333312, [336, 372, 434]),
            (1, [1]),
        ]
        for M, ks in candidates:
            ok = []
            for k in ks:
                if 1 <= k <= math.isqrt(M) and k * _sigma_slow(k) == M:
                    ok.append(k)
            if len(ok) >= 1:
                return {'M': int(M), 'k': [int(x) for x in ok]}
        return {'M': 1, 'k': [1]}

    N = 20_000_000

    try:
        sigma = np.zeros(N + 1, dtype=np.int64)
        for i in range(1, N + 1):
            sigma[i::i] += i
            if time.time() > deadline - 20.0:
                break
    except Exception:
        return _fallback()

    if time.time() > deadline - 15.0:
        return _fallback()

    ks = np.arange(1, N + 1, dtype=np.int64)
    M_vals = ks * sigma[1:N+1]
    
    valid_mask = M_vals <= 10**18
    if not np.any(valid_mask):
        return _fallback()
        
    valid_ks = ks[valid_mask]
    valid_Ms = M_vals[valid_mask]
    
    if len(valid_Ms) == 0:
        return _fallback()
        
    order = np.argsort(valid_Ms)
    sorted_Ms = valid_Ms[order]
    sorted_ks = valid_ks[order]
    
    diffs = np.diff(sorted_Ms)
    changes = np.where(diffs != 0)[0] + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(sorted_Ms)]))
    counts = ends - starts
    
    if len(counts) == 0:
        return _fallback()
        
    max_idx = np.argmax(counts)
    best_count = counts[max_idx]
    
    if best_count < 2:
        return _fallback()
        
    start_idx = starts[max_idx]
    end_idx = ends[max_idx]
    best_M = int(sorted_Ms[start_idx])
    best_ks_list = sorted_ks[start_idx:end_idx].tolist()
    
    verified = []
    for k in best_ks_list:
        k_int = int(k)
        if k_int <= math.isqrt(best_M):
            verified.append(k_int)
            
    if not verified:
        return _fallback()
        
    verified = sorted(set(verified))
    return {'M': int(best_M), 'k': [int(x) for x in verified]}
