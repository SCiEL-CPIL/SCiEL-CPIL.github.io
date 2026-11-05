import time
from math import isqrt

def construct():
    start_time = time.time()
    timeout = 115  # safety margin before 120s
    
    # Target M - increased to explore larger N
    M = 80_000_000
    
    powerful = set()
    max_y = int(M ** (1/3)) + 1
    
    for y in range(1, max_y + 1):
        y3 = y ** 3
        if y3 > M:
            break
        max_x = isqrt(M // y3)
        for x in range(1, max_x + 1):
            p = x * x * y3
            powerful.add(p)
            
    powerful_list = sorted(powerful)
    powerful_set = set(powerful_list)
    n_list = len(powerful_list)
    
    # Count sums using dictionary
    counts = {}
    
    for i in range(n_list):
        a = powerful_list[i]
        if time.time() - start_time > timeout - 15:
            break
        for j in range(i, n_list):
            b = powerful_list[j]
            s = a + b
            counts[s] = counts.get(s, 0) + 1
    
    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    best_N = max(counts, key=counts.get)
    
    # Reconstruct pairs
    pairs = []
    for a in powerful_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
