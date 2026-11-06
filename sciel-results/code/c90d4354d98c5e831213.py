import time
import random
import numpy as np

def construct():
    d = 7
    n_vertices = 1 << d  # 128
    num_edges = n_vertices * d  # 896
    
    # Precompute all edges and their witnesses
    # Edge is represented by (u, v) with u < v
    all_edges = []
    edge_to_idx = {}
    idx_to_edge = {}
    
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(all_edges)
                all_edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                idx_to_edge[idx] = (u, v)
                
    num_edges = len(all_edges)  # 448
    
    # Precompute witnesses for each edge
    # witnesses[i] is a list of lists of 3 edge indices
    witnesses = [[] for _ in range(num_edges)]
    
    for idx, (u, v) in enumerate(all_edges):
        # Find bit i
        diff = u ^ v
        i = diff.bit_length() - 1
        
        for j in range(d):
            if j == i:
                continue
            e_j = 1 << j
            u_j = u ^ e_j
            v_j = v ^ e_j
            
            # Edge 1: (u, u_j)
            e1 = (min(u, u_j), max(u, u_j))
            # Edge 2: (v, v_j)
            e2 = (min(v, v_j), max(v, v_j))
            # Edge 3: (u_j, v_j)
            e3 = (min(u_j, v_j), max(u_j, v_j))
            
            w1 = edge_to_idx[e1]
            w2 = edge_to_idx[e2]
            w3 = edge_to_idx[e3]
            
            witnesses[idx].append([w1, w2, w3])
    
    # Build conflict graph: edge a conflicts with edge b if they are in the same C4
    # Actually, we need: adding edge e creates a C4 if all 3 witnesses are present.
    # For local search, we need to know which edges conflict with which.
    
    def is_valid(S, e_idx):
        """Check if adding edge e_idx to set S of edge indices creates a C4."""
        w_list = witnesses[e_idx]
        for w in w_list:
            if w[0] in S and w[1] in S and w[2] in S:
                return False
        return True
    
    def greedy_fill(S, order):
        """Add all valid edges from order to S (copy), return new set."""
        S_new = set(S)
        for e in order:
            if e in S_new:
                continue
            if is_valid(S_new, e):
                S_new.add(e)
        return S_new
    
    def local_search(S, deadline, rng):
        """Perform 1-move local search with greedy fill. Returns improved set."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = list(S)
            rng.shuffle(S_list)
            
            for e_remove in S_list:
                if time.time() > deadline:
                    break
                
                S_without = set(S)
                S_without.discard(e_remove)
                
                # Greedy fill with random order
                candidates = [e for e in range(num_edges) if e not in S_without]
                rng.shuffle(candidates)
                S_new = greedy_fill(S_without, candidates)
                
                if len(S_new) > len(S):
                    S = S_new
                    improved = True
                    break
                elif len(S_new) == len(S) and random.random() < 0.2:
                    S = S_new
                    improved = True
                    break
        
        return S
    
    best_set = set()
    best_size = 0
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    rng = random.Random(42)
    
    # Multiple restarts with different strategies
    while time.time() < deadline:
        # Strategy 1: Random greedy
        S = set()
        order = list(range(num_edges))
        rng.shuffle(order)
        
        for e in order:
            if is_valid(S, e):
                S.add(e)
        
        # Local search
        S = local_search(S, deadline, rng)
        
        if len(S) > best_size:
            best_set = S.copy()
            best_size = len(S)
        
        # Strategy 2: Degree-based greedy (prefer edges with fewer conflicts)
        if time.time() < deadline:
            # Compute for each edge how many witnesses are "available"
            # Simpler: just try a few more random restarts
            S2 = set()
            order2 = list(range(num_edges))
            rng.shuffle(order2)
            
            for e in order2:
                if is_valid(S2, e):
                    S2.add(e)
            
            S2 = local_search(S2, deadline, rng)
            
            if len(S2) > best_size:
                best_set = S2.copy()
                best_size = len(S2)
        
        # Strategy 3: Start from best_set, remove random edge, refill
        if time.time() < deadline and best_size > 0:
            S3 = set(best_set)
            # Remove 1-3 random edges
            remove_count = rng.randint(1, 3)
            to_remove = rng.sample(list(S3), min(remove_count, len(S3)))
            for e in to_remove:
                S3.discard(e)
            
            # Greedy fill
            candidates = [e for e in range(num_edges) if e not in S3]
            rng.shuffle(candidates)
            S3 = greedy_fill(S3, candidates)
            
            # Local search
            S3 = local_search(S3, deadline, rng)
            
            if len(S3) > best_size:
                best_set = S3.copy()
                best_size = len(S3)
    
    # Convert best_set (indices) back to edges [u, v]
    result = []
    for idx in best_set:
        u, v = idx_to_edge[idx]
        result.append([u, v])
        
    return result
