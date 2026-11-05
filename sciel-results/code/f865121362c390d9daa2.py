import numpy as np
import random
import math
import time

def construct():
    n = 20
    
    # Precompute triples
    triples = []
    triple_index = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triples.append((i, j, k))
                triple_index[(i, j, k)] = idx
                idx += 1
    
    num_triples = len(triples)
    
    # Map vertex to list of (other1, other2, triple_idx) for triples containing it
    triples_with = [[] for _ in range(n)]
    for t_idx, (i, j, k) in enumerate(triples):
        triples_with[i].append((j, k, t_idx))
        triples_with[j].append((i, k, t_idx))
        triples_with[k].append((i, j, t_idx))
        
    def calc_pen(c):
        if c > 2:
            return (c - 2) ** 2
        return 0

    def run_sa(seed, time_limit=10.0):
        random.seed(seed)
        
        # Initialize with random graph
        adj_bits = [0] * n
        edge_count = 0
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.4:
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    edge_count += 1
        
        # Initialize counts
        counts = np.zeros(num_triples, dtype=np.int32)
        penalty = 0
        for t_idx, (i, j, k) in enumerate(triples):
            c = 0
            for m in range(n):
                if ((adj_bits[i] >> m) & 1) and ((adj_bits[j] >> m) & 1) and ((adj_bits[k] >> m) & 1):
                    c += 1
            counts[t_idx] = c
            penalty += calc_pen(c)
            
        best_edges = edge_count if penalty == 0 else -1
        best_adj_bits = adj_bits[:]
        
        T = 50.0
        cooling = 0.999
        min_T = 0.01
        Lambda = 100.0
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit and T > min_T:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            is_add = not ((adj_bits[u] >> v) & 1)
            
            delta_pen = 0
            
            # Update triples containing u
            for x, y, t_idx in triples_with[u]:
                if x == v or y == v:
                    continue
                if ((adj_bits[x] >> v) & 1) and ((adj_bits[y] >> v) & 1):
                    old_c = counts[t_idx]
                    new_c = old_c + (1 if is_add else -1)
                    delta_pen += calc_pen(new_c) - calc_pen(old_c)
                    counts[t_idx] = new_c
                    
            # Update triples containing v
            for x, y, t_idx in triples_with[v]:
                if x == u or y == u:
                    continue
                if ((adj_bits[x] >> u) & 1) and ((adj_bits[y] >> u) & 1):
                    old_c = counts[t_idx]
                    new_c = old_c + (1 if is_add else -1)
                    delta_pen += calc_pen(new_c) - calc_pen(old_c)
                    counts[t_idx] = new_c
                    
            delta_edges = 1 if is_add else -1
            delta_E = Lambda * delta_pen - delta_edges
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                if is_add:
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    edge_count += 1
                else:
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                    edge_count -= 1
                penalty += delta_pen
                
                if penalty == 0 and edge_count > best_edges:
                    best_edges = edge_count
                    best_adj_bits = adj_bits[:]
                    
            T *= cooling
            
        return best_edges, best_adj_bits

    best_edges_overall = 0
    best_adj_overall = None
    
    start = time.time()
    seed = 42
    while time.time() - start < 115.0:
        edges, adj_bits = run_sa(seed, time_limit=10.0)
        if edges > best_edges_overall:
            best_edges_overall = edges
            best_adj_overall = adj_bits
        seed += 1
        
    if best_adj_overall is None:
        return []
        
    edges_list = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj_overall[u] >> v) & 1:
                edges_list.append([u, v])
                
    return edges_list
