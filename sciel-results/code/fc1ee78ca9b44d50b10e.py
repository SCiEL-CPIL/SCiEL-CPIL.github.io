import time

N = 18

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    comps = batcher_sort(N, 0)
    
    return comps

def batcher_sort(n, offset):
    if n <= 1:
        return []
    m = n // 2
    comps = []
    comps += batcher_sort(m, offset)
    comps += batcher_sort(n - m, offset + m)
    comps += batcher_merge(m, n - m, offset)
    return comps

def batcher_merge(n, m, offset):
    if n == 0 or m == 0:
        return []
    if n == 1 and m == 1:
        return [(offset, offset + 1)]
    if n == 1:
        return [(offset, offset + 1)] + batcher_merge(1, m - 1, offset + 1)
    if m == 1:
        return [(offset + n - 1, offset + n)] + batcher_merge(n - 1, 1, offset)
    
    N_total = n + m
    K = N_total // 2
    comps = []
    comps += batcher_merge(n // 2, m // 2, offset)
    comps += batcher_merge(n - n // 2, m - m // 2, offset + K)
    
    # Cross comparators
    # Even
    for i in range(0, K, 2):
        comps.append((offset + i, offset + i + K))
    # Odd
    for i in range(1, K, 2):
        comps.append((offset + i, offset + i + K))
        
    return comps
