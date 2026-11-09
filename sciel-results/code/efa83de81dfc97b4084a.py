import math
from collections import defaultdict

def construct():
    import time
    start_time = time.time()
    DEADLINE = 115.0  # seconds, leaving 5s safety margin

    best_result = {'M': 333312, 'k': [336, 372, 434]}
    best_count = 3

    try:
        import numpy as np
        use_numpy = True
    except ImportError:
        use_numpy = False

    if use_numpy:
        max_k = 4000000  # 4 million
        sigma = np.zeros(max_k + 1, dtype=np.int64)
        for i in range(1, max_k + 1):
            sigma[i::i] += i
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            return best_result

        M_to_ks = defaultdict(list)
        k_arr = np.arange(1, max_k + 1, dtype=np.int64)
        M_arr = k_arr * sigma[1:max_k + 1]
        valid = (M_arr <= 10**18) & (k_arr <= sigma[1:max_k + 1])
        
        for idx in range(max_k):
            if valid[idx]:
                k = int(k_arr[idx])
                M = int(M_arr[idx])
                M_to_ks[M].append(k)
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            return best_result

        for M, ks in M_to_ks.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_result = {'M': int(M), 'k': [int(x) for x in ks[:1000]]}

    else:
        max_k = 1000000
        sigma = [0] * (max_k + 1)
        for i in range(1, max_k + 1):
            for j in range(i, max_k + 1, i):
                sigma[j] += i
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            return best_result

        M_to_ks = defaultdict(list)
        for k in range(1, max_k + 1):
            M = k * sigma[k]
            if M <= 10**18 and k <= sigma[k]:
                M_to_ks[M].append(k)
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            return best_result

        for M, ks in M_to_ks.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_result = {'M': int(M), 'k': [int(x) for x in ks[:1000]]}

    return best_result
