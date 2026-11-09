import time
import random
import sys

def construct():
    n = 31
    target_edges = 91
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_c4_count = float('inf')
    best_edges = None
    
    # We try to construct a 90-edge C4-free graph, then add the best edge.
    # We use randomized greedy to build the C4-free base.
    
    for seed in range(10000):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        
        # 1. Build 90-edge C4-free graph
        edges = set()
        neighbors = [[] for _ in range(n)]
        
        # Pre-compute all pairs
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        
        # Greedy add edges
        for u, v in all_pairs:
            if len(edges) >= 90:
                break
            
            # Check if adding (u, v) creates a C4
            # A C4 is created if there is a path u-a-b-v of length 3
            # This is equivalent to: exists a in N(u) such that N(a) intersects N(v) 
            # AND that intersection vertex b is not u or v?
            # Actually, path u-a-b-v means:
            # a is neighbor of u.
            # b is neighbor of v.
            # a is neighbor of b.
            # So we check if there is a common neighbor between N(u) and N(v) is NOT the condition.
            # The condition is: | { a in N(u) | N(a) intersects N(v) } | > 0 ?
            # No.
            # Path u-a-b-v.
            # a ~ u
            # b ~ v
            # a ~ b
            # So for each a in N(u), check if any neighbor of a is in N(v).
            
            creates_c4 = False
            for a in neighbors[u]:
                # Check if a is connected to any neighbor of v
                # We can do this by iterating N(a) and checking if in N(v)
                # Or iterate N(v) and check if in N(a)
                # Since degrees are small, either is fine.
                for b in neighbors[v]:
                    if a == b:
                        # Triangle u-a-v, not a C4
                        continue
                    if b in neighbors[a]:
                        creates_c4 = True
                        break
                if creates_c4:
                    break
            
            if not creates_c4:
                edges.add((u, v))
                neighbors[u].append(v)
                neighbors[v].append(u)
        
        if len(edges) < 90:
            continue
            
        # 2. Find best edge to add
        # We have a C4-free graph with 90 edges.
        # Calculate C4 count for adding each non-edge.
        
        current_min_c4 = float('inf')
        best_add_edge = None
        
        for u in range(n):
            for v in range(u+1, n):
                if (u, v) in edges:
                    continue
                # Count length-3 paths between u and v
                # Path u-a-b-v
                count = 0
                for a in neighbors[u]:
                    if a == v: continue
                    for b in neighbors[v]:
                        if b == u: continue
                        if a == b: continue
                        if b in neighbors[a]:
                            count += 1
                            
                if count < current_min_c4:
                    current_min_c4 = count
                    best_add_edge = (u, v)
                    
        # The new graph has current_min_c4 C4s.
        if current_min_c4 < best_c4_count:
            best_c4_count = current_min_c4
            best_edges = list(edges) + [list(best_add_edge)]
            
            if best_c4_count == 1:
                # Can't get better than 1
                break

    if best_edges is None:
        # Fallback: return seed-like structure or random
        # This should not happen
        return [[i, j] for i in range(31) for j in range(i+1, 31)][:91]
        
    return best_edges
