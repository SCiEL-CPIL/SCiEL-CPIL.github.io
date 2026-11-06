import time
import random

N = 9
DEADLINE = 145

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    bit_masks = [1 << b for b in range(N)]
    rng = random.Random(12345)
    
    # Precompute neighbors
    neighbors = [[] for _ in range(2**N)]
    for v in range(2**N):
        for b in range(N):
            neighbors[v].append(v ^ bit_masks[b])
    
    best_path = None
    best_len = 0
    
    all_vertices = list(range(2**N))
    rng.shuffle(all_vertices)
    
    for idx, start_v in enumerate(all_vertices):
        if time.time() > deadline - 2:
            break
        
        # Use a different random seed for each start
        local_rng = random.Random(idx * 7919 + 1)
        result = _dfs_snake(start_v, deadline, bit_masks, neighbors, local_rng)
        if result and len(result) > best_len:
            best_len = len(result)
            best_path = result
    
    if best_path is None:
        best_path = _greedy_snake()
    
    return best_path

def _greedy_snake():
    path = [0]
    used = {0}
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            if any((v ^ p).bit_count() == 1 for p in path[:-1]):
                continue
            nxt = v
            break
        if nxt is None:
            return path
        path.append(nxt)
        used.add(nxt)

def _dfs_snake(start_v, deadline, bit_masks, neighbors, rng):
    path = [start_v]
    used = {start_v}
    adj_count = [0] * (2**N)
    best = list(path)
    
    def is_valid(v):
        return v not in used and adj_count[v] == 0
    
    def add_vertex(v):
        old_last = path[-1]
        for u in neighbors[old_last]:
            adj_count[u] += 1
        path.append(v)
        used.add(v)
    
    def remove_vertex():
        v = path.pop()
        used.discard(v)
        if path:
            new_last = path[-1]
            for u in neighbors[new_last]:
                adj_count[u] -= 1
    
    def dfs():
        nonlocal best
        if time.time() > deadline - 1:
            return
        
        if len(path) > len(best):
            best = list(path)
        
        cur = path[-1]
        bits = list(range(N))
        rng.shuffle(bits)
        
        for b in bits:
            v = cur ^ bit_masks[b]
            if is_valid(v):
                add_vertex(v)
                dfs()
                remove_vertex()
                if time.time() > deadline - 1:
                    return
    
    dfs()
    return best
