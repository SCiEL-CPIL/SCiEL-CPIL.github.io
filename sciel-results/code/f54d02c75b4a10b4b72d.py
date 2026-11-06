import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 16
    all_8sets = list(combinations(range(n), 8))
    
    def is_q3(vs, adj_bits):
        local_adj = [0]*8
        edge_count = 0
        for i in range(8):
            u = vs[i]
            mask = adj_bits[u]
            for j in range(i+1, 8):
                v = vs[j]
                if (mask >> v) & 1:
                    local_adj[i] |= (1 << j)
                    local_adj[j] |= (1 << i)
                    edge_count += 1
        
        if edge_count != 12:
            return False
            
        for i in range(8):
            if bin(local_adj[i]).count('1') != 3:
                return False
                
        color = [-1]*8
        color[0] = 0
        stack = [0]
        while stack:
            u = stack.pop()
            cu = color[u]
            mask = local_adj[u]
            while mask:
                lsb = mask & -mask
                v = (lsb.bit_length() - 1)
                mask ^= lsb
                if color[v] == -1:
                    color[v] = 1 - cu
                    stack.append(v)
                elif color[v] == cu:
                    return False
        return True

    def find_one_q3(adj_bits):
        for vs in all_8sets:
            if is_q3(vs, adj_bits):
                return vs
        return None

    def edge_creates_q3(u, v, adj_bits):
        others = [x for x in range(n) if x != u and x != v]
        for rest in combinations(others, 6):
            vs = tuple(sorted((u, v) + rest))
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            
            if is_q3(vs, adj_bits):
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
                return True
                
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            
        return False

    best_edges = set()
    best_count = 0
    
    for seed in range(1000):
        if time.time() > deadline - 5:
            break
        random.seed(seed)
        
        edges = set((i, j) for i in range(n) for j in range(i+1, n))
        adj_bits = [0]*n
        for (u, v) in edges:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            
        while True:
            if time.time() > deadline - 5:
                break
            vs = find_one_q3(adj_bits)
            if not vs:
                break
            local_edges = []
            for i in range(8):
                for j in range(i+1, 8):
                    if (adj_bits[vs[i]] >> vs[j]) & 1:
                        local_edges.append((vs[i], vs[j]))
            rem_u, rem_v = random.choice(local_edges)
            if rem_u > rem_v:
                rem_u, rem_v = rem_v, rem_u
            edges.remove((rem_u, rem_v))
            adj_bits[rem_u] &= ~(1 << rem_v)
            adj_bits[rem_v] &= ~(1 << rem_u)
            
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edges]
        random.shuffle(missing)
        
        for (u, v) in missing:
            if time.time() > deadline - 5:
                break
            if not edge_creates_q3(u, v, adj_bits):
                edges.add((u, v))
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)
            
    return [[u, v] for (u, v) in best_edges]
