# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    if limit < 1:
        return [1]
    powerful = set()
    powerful.add(1)
    max_b = int(limit ** (1/3)) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return sorted(powerful)

def construct():
    start = time.time()
    deadline = start + 100
    
    limit = 50_000_000
    powerful = _generate_powerful(limit)
    n = len(powerful)
    
    if time.time() > deadline:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    p_arr = np.array(powerful, dtype=np.int64)
    p_set = set(powerful)
    
    best_N = 2
    best_count = 1
    best_pairs = [[1, 1]]
    
    # Use numpy to vectorize sum generation in chunks
    # For each chunk of a's, compute all valid sums
    chunk_size = 1000
    sum_dict = {}
    
    for i in range(0, n, chunk_size):
        if time.time() > deadline:
            break
        end = min(i + chunk_size, n)
        chunk = p_arr[i:end]
        
        # For each a in chunk, valid b range is [a, 2*limit - a]
        # We'll compute sums vectorized per a
        for a in chunk:
            max_b = 2 * limit - a
            if max_b < a:
                continue
            mask = (p_arr >= a) & (p_arr <= max_b)
            valid_b = p_arr[mask]
            if len(valid_b) == 0:
                continue
            sums = a + valid_b
            sums = sums[sums <= 10**18]
            for s in sums:
                s = int(s)
                if s not in sum_dict:
                    sum_dict[s] = []
                sum_dict[s].append((a, s - a))
    
    for N, pairs in sum_dict.items():
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_N = N
            best_pairs = pairs
    
    final_pairs = [[a, b] for a, b in best_pairs if a <= b and a + b == best_N and a in p_set and b in p_set]
    if not final_pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N, 'pairs': final_pairs}
# EVOLVE-BLOCK-END
