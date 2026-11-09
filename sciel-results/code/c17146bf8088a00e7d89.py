import time
import math
import sys

def construct():
    t0 = time.time()
    DEADLINE = 110.0
    MIN_M, MAX_M = 20, 200
    
    # 1. Generate nodes
    nodes = []
    node_index = {}
    idx = 0
    for m in range(MIN_M, MAX_M + 1):
        for r in range(m):
            node_index[(m, r)] = idx
            nodes.append((m, r))
            idx += 1
            
    N = len(nodes)
    
    # 2. Build conflict bitsets
    conflicts = [0] * N
    
    # Same modulus conflicts
    for m in range(MIN_M, MAX_M + 1):
        indices = [node_index[(m, r)] for r in range(m)]
        mask = 0
        for i in indices:
            mask |= (1 << i)
        for i in indices:
            conflicts[i] |= (mask ^ (1 << i))
            
    # Different modulus conflicts
    for m1 in range(MIN_M, MAX_M + 1):
        for m2 in range(m1 + 1, MAX_M + 1):
            g = math.gcd(m1, m2)
            for r1 in range(m1):
                i1 = node_index[(m1, r1)]
                rem = r1 % g
                for r2 in range(rem, m2, g):
                    i2 = node_index[(m2, r2)]
                    conflicts[i1] |= (1 << i2)
                    conflicts[i2] |= (1 << i1)
                    
    # 3. Prepare for B&B
    weights = [1.0 / nodes[i][0] for i in range(N)]
    
    modulus_mask = {}
    for m in range(MIN_M, MAX_M + 1):
        mask = 0
        for r in range(m):
            mask |= (1 << node_index[(m, r)])
        modulus_mask[m] = mask
        
    best_score = 0.0
    best_set = []
    
    sorted_indices = sorted(range(N), key=lambda i: nodes[i][0])
    
    def get_next_node(available):
        for i in sorted_indices:
            if available & (1 << i):
                return i
        return -1

    def upper_bound(available):
        bound = 0.0
        for m in range(MIN_M, MAX_M + 1):
            if available & modulus_mask[m]:
                bound += 1.0 / m
        return bound

    sys.setrecursionlimit(10000)
    
    def solve(available, current_score, current_set):
        nonlocal best_score, best_set
        
        if time.time() > DEADLINE:
            return
            
        if available == 0:
            if current_score > best_score:
                best_score = current_score
                best_set = current_set[:]
            return
            
        bound = current_score + upper_bound(available)
        if bound <= best_score:
            return
            
        v = get_next_node(available)
        if v == -1:
            return
            
        new_avail = available & ~(conflicts[v] | (1 << v))
        current_set.append(v)
        solve(new_avail, current_score + weights[v], current_set)
        current_set.pop()
        
        new_avail2 = available & ~(1 << v)
        solve(new_avail2, current_score, current_set)
        
    initial_available = (1 << N) - 1
    solve(initial_available, 0.0, [])
    
    result = []
    for i in best_set:
        m, r = nodes[i]
        result.append([m, r])
        
    return result
