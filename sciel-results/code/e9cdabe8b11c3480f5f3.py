import time
import random
from collections import deque

N = 12
DEADLINE = 85  # safety margin before 90s kill

def _bfs_reset_length(perm_a, perm_b):
    """Compute exact shortest reset word length using bitmask BFS with precomputed transitions."""
    size = 1 << N
    trans_a = [0] * size
    trans_b = [0] * size
    
    for mask in range(size):
        result_a = 0
        result_b = 0
        for i in range(N):
            if mask & (1 << i):
                result_a |= (1 << perm_a[i])
                result_b |= (1 << perm_b[i])
        trans_a[mask] = result_a
        trans_b[mask] = result_b
    
    full_mask = size - 1
    visited = {full_mask: 0}
    queue = deque([full_mask])
    
    while queue:
        mask = queue.popleft()
        dist = visited[mask]
        
        if mask & (mask - 1) == 0:
            return dist
        
        next_a = trans_a[mask]
        next_b = trans_b[mask]
        
        if next_a not in visited:
            visited[next_a] = dist + 1
            queue.append(next_a)
        if next_b not in visited:
            visited[next_b] = dist + 1
            queue.append(next_b)
    
    return -1

def construct():
    start_time = time.time()
    
    best_threshold = 0
    best_a = None
    best_b = None
    
    def _update(a, b):
        nonlocal best_threshold, best_a, best_b
        thresh = _bfs_reset_length(a, b)
        if thresh > best_threshold:
            best_threshold = thresh
            best_a = a[:]
            best_b = b[:]
    
    # Start with Černý-like
    a_cycle = [(i + 1) % N for i in range(N)]
    b_fold = [0] + [i - 1 for i in range(1, N)]
    _update(a_cycle, b_fold)
    
    if time.time() - start_time > DEADLINE:
        return [best_a, best_b]
    
    # Random search with good initialization
    random.seed(42)
    for _ in range(200000):
        if time.time() - start_time > DEADLINE:
            break
        a = list(range(N))
        random.shuffle(a)
        b = [0] * N
        for i in range(1, N):
            b[i] = random.randint(0, i)
        _update(a, b)
    
    if best_a is None:
        best_a = a_cycle
        best_b = b_fold
    
    return [best_a, best_b]
