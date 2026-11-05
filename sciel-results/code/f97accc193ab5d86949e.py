import time
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    timeout = 115
    
    M = 2_000_000_000
    
    powerful = set()
    max_y = int(M ** (1/3)) + 1
    
    for y in range(1, max_y + 1):
        y3 = y ** 3
        if y3 > M:
            break
        max_x = isqrt(M // y3)
        for x in range(1, max_x + 1):
            p = x * x * y3
            if p <= M:
                powerful.add(p)
                
        if y % 50 == 0 and time.time() - start_time > timeout - 30:
            break

    powerful_list = sorted(powerful)
    n_list = len(powerful_list)
    
    counts = defaultdict(int)
    pl = powerful_list
    cnt = counts
    K = n_list
    
    time_check_interval = 500
    
    for i in range(K):
        if i % time_check_interval == 0:
            if time.time() - start_time > timeout - 20:
                break
        a = pl[i]
        for j in range(i, K):
            s = a + pl[j]
            cnt[s] += 1
            
    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    best_N = max(counts, key=counts.get)
    
    powerful_set = set(powerful_list)
    pairs = []
    for a in powerful_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
