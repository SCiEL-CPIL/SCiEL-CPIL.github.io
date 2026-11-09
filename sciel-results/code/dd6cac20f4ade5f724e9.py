import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # 1. Build Conflict Graph
    adj = [0] * (N + 1)
    is_square = set(k*k for k in range(1, 23))
    
    for u in vertices:
        for v in vertices:
            if u < v:
                if (u + v) in is_square:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
    valid_vertices = [u for u in vertices if (2*u) not in is_square]
    
    initial_available = 0
    for u in valid_vertices:
        initial_available |= (1 << u)
        
    degs = []
    for u in valid_vertices:
        degs.append((bin(adj[u] & initial_available).count('1'), u))
    degs.sort(reverse=True)
    search_order = [u for _, u in degs]
    
    # Multiple Greedy Initializations
    best_size = 0
    best_set = []
    
    def greedy_solution(avail_mask, randomize=False):
        s = set()
        mask = avail_mask
        while mask:
            candidates = []
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = bin(adj[v] & mask).count('1')
                candidates.append((deg, v))
                m ^= lsb
                
            if not candidates: break
            
            if randomize:
                # Pick from top 3 degrees randomly
                candidates.sort()
                k = min(3, len(candidates))
                best_v = random.choice(candidates[:k])[1]
            else:
                best_v = min(candidates)[1]
                
            s.add(best_v)
            mask &= ~(1 << best_v)
            mask &= ~adj[best_v]
        return s

    # Run greedy a few times
    for i in range(10):
        if time.time() > deadline - 5:
            break
        s = greedy_solution(initial_available, randomize=(i > 0))
        if len(s) > best_size:
            best_size = len(s)
            best_set = list(s)
            
    global_best_size = best_size
    global_best_set = best_set
    
    def popcount(x):
        return bin(x).count('1')
        
    def dfs(available, current_size, current_set):
        nonlocal global_best_size, global_best_set
        
        if time.time() > deadline:
            return
            
        if current_size > global_best_size:
            global_best_size = current_size
            global_best_set = list(current_set)
            
        if available == 0:
            return
            
        if current_size + popcount(available) <= global_best_size:
            return
            
        v = -1
        for u in search_order:
            if available & (1 << u):
                v = u
                break
                
        if v == -1:
            return
            
        bit_v = 1 << v
        
        new_avail_incl = available & ~bit_v & ~adj[v]
        current_set.append(v)
        dfs(new_avail_incl, current_size + 1, current_set)
        current_set.pop()
        
        new_avail_excl = available & ~bit_v
        dfs(new_avail_excl, current_size, current_set)
        
    import sys
    sys.setrecursionlimit(5000)
    
    dfs(initial_available, 0, [])
    
    # Verify
    valid = True
    for i in range(len(global_best_set)):
        for j in range(i, len(global_best_set)):
            s = global_best_set[i] + global_best_set[j]
            r = int(math.sqrt(s))
            if r * r == s:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        global_best_set = list(best_set)
        
    return global_best_set
