import numpy as np
import random

def construct():
    d = 7
    n = 2 ** d  # 128 vertices
    
    # Generate all edges of Q7
    edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
                edge_to_idx[(y, x)] = idx
    
    m = len(edges)  # 448
    
    # Build conflict graph
    conflict = [set() for _ in range(m)]
    
    by_dir = [[] for _ in range(d)]
    for idx, (x, y) in enumerate(edges):
        diff = x ^ y
        dir_bit = diff.bit_length() - 1
        by_dir[dir_bit].append(idx)
    
    for dir_bit in range(d):
        dir_edges = by_dir[dir_bit]
        for i in range(len(dir_edges)):
            for j in range(i + 1, len(dir_edges)):
                idx1 = dir_edges[i]
                idx2 = dir_edges[j]
                x1, y1 = edges[idx1]
                x2, y2 = edges[idx2]
                diff12 = x1 ^ x2
                if diff12 & (1 << dir_bit) == 0 and diff12.bit_count() == 1:
                    conflict[idx1].add(idx2)
                    conflict[idx2].add(idx1)
    
    # Precompute neighbor lists for faster access
    neighbors = [list(c) for c in conflict]
    
    def get_mcs_order(random_seed):
        random.seed(random_seed)
        labels = [0] * m
        visited = [False] * m
        order = []
        
        # Start with a random vertex
        start = random.randint(0, m - 1)
        visited[start] = True
        order.append(start)
        
        for nb in neighbors[start]:
            if not visited[nb]:
                labels[nb] += 1
                
        for _ in range(m - 1):
            # Find unvisited vertex with max label
            max_label = -1
            best_v = -1
            for v in range(m):
                if not visited[v] and labels[v] > max_label:
                    max_label = labels[v]
                    best_v = v
            
            if best_v == -1:
                # Fallback (should not happen for connected graph, but just in case)
                for v in range(m):
                    if not visited[v]:
                        best_v = v
                        break
            
            visited[best_v] = True
            order.append(best_v)
            
            for nb in neighbors[best_v]:
                if not visited[nb]:
                    labels[nb] += 1
                    
        return order

    def greedy_solve(random_seed):
        mcs_order = get_mcs_order(random_seed)
        # Create a mapping from vertex to its index in mcs_order
        mcs_rank = [0] * m
        for i, v in enumerate(mcs_order):
            mcs_rank[v] = i
            
        remaining = set(range(m))
        selected = []
        
        while remaining:
            min_deg = float('inf')
            best_candidates = []
            
            for idx in remaining:
                deg = len(conflict[idx] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_candidates = [idx]
                elif deg == min_deg:
                    best_candidates.append(idx)
            
            # Pick the candidate with the smallest MCS rank (earliest in MCS order)
            # This prioritizes vertices that are "central" in the MCS sense
            best = min(best_candidates, key=lambda x: mcs_rank[x])
            
            selected.append(best)
            remaining.remove(best)
            
            for c in conflict[best]:
                if c in remaining:
                    remaining.remove(c)
                    
        return selected

    # Run multiple iterations with different random seeds
    best_set = []
    best_size = 0
    
    # Estimate time: m=448. MCS is O(m^2). Greedy is O(m^2).
    # 448^2 ~ 200k. Python does ~10M ops/s? So 0.02s per run.
    # We can do ~1000 runs in 20s. Let's do 500 to be safe.
    
    for i in range(500):
        current_set = greedy_solve(i)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    result = []
    for idx in best_set:
        x, y = edges[idx]
        result.append([x, y])
        
    return result
