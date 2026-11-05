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
    
    # Map vertex to list of triple indices containing it
    triples_with = [[] for _ in range(n)]
    for t_idx, (i, j, k) in enumerate(triples):
        triples_with[i].append(t_idx)
        triples_with[j].append(t_idx)
        triples_with[k].append(t_idx)
        
    # Precompute the "others" for each vertex in each triple it belongs to
    other_vertices = {}
    for t_idx, (i, j, k) in enumerate(triples):
        other_vertices[(t_idx, i)] = (j, k)
        other_vertices[(t_idx, j)] = (i, k)
        other_vertices[(t_idx, k)] = (i, j)

    def calc_pen(c):
        if c > 2:
            return (c - 2) ** 2
        return 0

    def compute_penalty_and_counts(adj):
        counts = np.zeros(num_triples, dtype=np.int32)
        total_pen = 0
        for t_idx, (i, j, k) in enumerate(triples):
            c = 0
            for m in range(n):
                if adj[i, m] and adj[j, m] and adj[k, m]:
                    c += 1
            counts[t_idx] = c
            total_pen += calc_pen(c)
        return counts, total_pen

    def count_edges(adj):
        return int(np.sum(adj) // 2)

    def run_sa(seed, time_limit):
        random.seed(seed)
        
        # Initialize graph
        # Try a few different densities to see which works better
        # p=0.5 is a good middle ground
        adj = np.zeros((n, n), dtype=np.uint8)
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.5:
                    adj[u, v] = 1
                    adj[v, u] = 1
        
        counts, penalty = compute_penalty_and_counts(adj)
        edge_count = count_edges(adj)
        
        best_edges = edge_count if penalty == 0 else 0
        best_adj = adj.copy() if penalty == 0 else None
        
        T = 50.0
        cooling = 0.995
        min_T = 0.01
        Lambda = 10.0
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit and T > min_T:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            is_add = not adj[u, v]
            
            delta_pen = 0
            
            # Update for triples containing u but not v
            for t_idx in triples_with[u]:
                i, j, k = triples[t_idx]
                if v == i or v == j or v == k:
                    continue
                x, y = other_vertices[(t_idx, u)]
                if is_add:
                    if adj[x, v] and adj[y, v]:
                        old_c = counts[t_idx]
                        new_c = old_c + 1
                        delta_pen += calc_pen(new_c) - calc_pen(old_c)
                        counts[t_idx] = new_c
                else:
                    if adj[x, v] and adj[y, v]:
                        old_c = counts[t_idx]
                        new_c = old_c - 1
                        delta_pen += calc_pen(new_c) - calc_pen(old_c)
                        counts[t_idx] = new_c
                        
            # Update for triples containing v but not u
            for t_idx in triples_with[v]:
                i, j, k = triples[t_idx]
                if u == i or u == j or u == k:
                    continue
                x, y = other_vertices[(t_idx, v)]
                if is_add:
                    if adj[x, u] and adj[y, u]:
                        old_c = counts[t_idx]
                        new_c = old_c + 1
                        delta_pen += calc_pen(new_c) - calc_pen(old_c)
                        counts[t_idx] = new_c
                else:
                    if adj[x, u] and adj[y, u]:
                        old_c = counts[t_idx]
                        new_c = old_c - 1
                        delta_pen += calc_pen(new_c) - calc_pen(old_c)
                        counts[t_idx] = new_c
                        
            delta_edges = 1 if is_add else -1
            delta_E = -delta_edges + Lambda * delta_pen
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                adj[u, v] = 1 if is_add else 0
                adj[v, u] = 1 if is_add else 0
                edge_count += delta_edges
                penalty += delta_pen
                
                if penalty == 0 and edge_count > best_edges:
                    best_edges = edge_count
                    best_adj = adj.copy()
            
            T *= cooling
            
        # If we didn't find any feasible solution, try to repair the current one quickly
        if best_adj is None and penalty > 0:
            # Greedy repair
            improved = True
            while improved and penalty > 0:
                improved = False
                best_delta = 0
                best_edge = None
                for u in range(n):
                    for v in range(u+1, n):
                        if adj[u, v]:
                            delta_pen = 0
                            for t_idx in triples_with[u]:
                                i, j, k = triples[t_idx]
                                if v == i or v == j or v == k: continue
                                x, y = other_vertices[(t_idx, u)]
                                if adj[x, v] and adj[y, v]:
                                    old_c = counts[t_idx]
                                    new_c = old_c - 1
                                    delta_pen += calc_pen(new_c) - calc_pen(old_c)
                                    counts[t_idx] = new_c
                            for t_idx in triples_with[v]:
                                i, j, k = triples[t_idx]
                                if u == i or u == j or u == k: continue
                                x, y = other_vertices[(t_idx, v)]
                                if adj[x, u] and adj[y, u]:
                                    old_c = counts[t_idx]
                                    new_c = old_c - 1
                                    delta_pen += calc_pen(new_c) - calc_pen(old_c)
                                    counts[t_idx] = new_c
                                    
                            if delta_pen < best_delta:
                                best_delta = delta_pen
                                best_edge = (u, v)
                                
                            # Undo
                            for t_idx in triples_with[u]:
                                i, j, k = triples[t_idx]
                                if v == i or v == j or v == k: continue
                                x, y = other_vertices[(t_idx, u)]
                                if adj[x, v] and adj[y, v]:
                                    counts[t_idx] += 1
                            for t_idx in triples_with[v]:
                                i, j, k = triples[t_idx]
                                if u == i or u == j or u == k: continue
                                x, y = other_vertices[(t_idx, v)]
                                if adj[x, u] and adj[y, u]:
                                    counts[t_idx] += 1
                                    
                if best_edge:
                    u, v = best_edge
                    adj[u, v] = 0
                    adj[v, u] = 0
                    edge_count -= 1
                    penalty += best_delta
                    improved = True
                    if penalty == 0 and edge_count > best_edges:
                        best_edges = edge_count
                        best_adj = adj.copy()
        
        if best_adj is None:
            # Fallback: return empty or a small valid graph
            return []
            
        return best_adj, best_edges

    total_time = 115.0
    num_runs = 5
    time_per_run = total_time / num_runs
    
    global_best_adj = None
    global_best_edges = 0
    
    for i in range(num_runs):
        seed = 12345 + i
        adj, edges = run_sa(seed, time_per_run)
        if edges > global_best_edges:
            global_best_edges = edges
            global_best_adj = adj
            
    if global_best_adj is None:
        return []
        
    edges_list = []
    for u in range(n):
        for v in range(u+1, n):
            if global_best_adj[u, v]:
                edges_list.append([u, v])
                
    return edges_list
