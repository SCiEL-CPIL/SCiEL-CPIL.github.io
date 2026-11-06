import time
import numpy as np
from math import isqrt
from collections import Counter

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    max_b = int(limit ** (1.0 / 3.0)) + 2
    is_sq_free = np.ones(max_b + 1, dtype=bool)
    is_sq_free[0] = False
    for i in range(2, isqrt(max_b) + 1):
        sq = i * i
        is_sq_free[sq::sq] = False
    is_sq_free[1] = True

    powerful = set()
    for b in range(1, max_b + 1):
        if not is_sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return powerful

def construct():
    t0 = time.time()
    deadline = t0 + 115  # 115s safety margin

    best_result = {'N': 2, 'pairs': [[1, 1]]}
    best_count = 1

    limits_to_try = [10**12, 10**13]

    for limit in limits_to_try:
        if time.time() > deadline - 30:
            break

        powerful = _generate_powerful(limit)
        n_powerful = len(powerful)
        
        if n_powerful < 10:
            continue

        powerful_list = sorted(powerful)
        powerful_set = powerful
        arr = np.array(powerful_list, dtype=np.int64)
        
        counter = Counter()
        
        block_size = 1000
        n = n_powerful
        for i in range(0, n, block_size):
            if time.time() > deadline - 15:
                break
            a_block = arr[i:i+block_size]
            sums = a_block[:, None] + arr[None, :]
            counter.update(sums.flatten())
            
        if counter:
            max_N, max_count = counter.most_common(1)[0]
            
            is_even = (max_N % 2 == 0)
            half = max_N // 2
            has_diag = 1 if (is_even and half in powerful_set) else 0
            
            valid_pairs_count = (max_count + has_diag) // 2
            
            if valid_pairs_count > best_count:
                best_count = valid_pairs_count
                pairs = []
                for a in powerful_list:
                    if 2 * a > max_N:
                        break
                    b = max_N - a
                    if b in powerful_set:
                        pairs.append([a, b])
                best_result = {'N': max_N, 'pairs': pairs}

    return best_result
