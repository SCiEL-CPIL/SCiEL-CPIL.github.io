import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 40
    target_edges = 401
    
    # Partition sizes
    # A: size 9 (vertices 0..8)
    # B: size 31 (vertices 9..39)
    a_size = 9
    b_size = 31
    
    A = list(range(a_size))
    B = list(range(a_size, n))
    
    edges = set()
    
    # 1. Add all edges between A and B
    for a in A:
        for b in B:
            edges.add((a, b))
            
    needed_in_B = target_edges - (a_size * b_size)
    # 122
    
    # Deterministic greedy for B-B edges
    # We want to find a triangle-free graph with 122 edges and max degree 8.
    
    B_edges = []
    degree = {v: 0 for v in B}
    adj = {v: set() for v in B}
    
    # Generate all pairs
    pairs = list(itertools.combinations(B, 2))
    
    # Use a priority queue approach implicitly by re-evaluating best candidate
    # To speed up, we maintain a list of candidates and pick the best one.
    # Since 465 pairs is small, O(E * Pairs) is fine.
    
    candidates = set(pairs)
    
    def try_build(max_deg):
        nonlocal B_edges, degree, adj, candidates
        B_edges = []
        degree = {v: 0 for v in B}
        adj = {v: set() for v in B}
        candidates = set(pairs)
        
        while len(B_edges) < needed_in_B:
            if time.time() > deadline:
                return False
                
            best_pair = None
            best_score = None
            
            # Iterate all remaining candidates
            for u, v in candidates:
                # Check triangle
                if adj[u] & adj[v]:
                    continue
                    
                # Check degree
                if degree[u] >= max_deg or degree[v] >= max_deg:
                    continue
                
                # Score: Prefer pairs where both degrees are low.
                # Tie-breaker: random or fixed order for determinism
                score = degree[u] + degree[v]
                
                if best_score is None or score < best_score:
                    best_score = score
                    best_pair = (u, v)
                    
            if best_pair is None:
                return False
                
            u, v = best_pair
            B_edges.append((u, v))
            degree[u] += 1
            degree[v] += 1
            adj[u].add(v)
            adj[v].add(u)
            candidates.remove((u, v))
            
        return True

    # Try to build with max degree 8
    if not try_build(8):
        # If failed, try max degree 9
        if not try_build(9):
             # Fallback: just fill randomly if stuck (should not happen)
             pass

    for u, v in B_edges:
        edges.add((u, v))
        
    # Verify we have exactly 401 edges
    if len(edges) != target_edges:
        # Fallback: add random edges if we are short
        all_pairs = list(itertools.combinations(range(n), 2))
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if len(edges) >= target_edges:
                break
            if (u, v) not in edges:
                edges.add((u, v))
                
    result = [[u, v] for u, v in edges]
    
    return result
