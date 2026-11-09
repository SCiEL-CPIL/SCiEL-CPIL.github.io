import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    target_edges = 401
    
    a_size = 9
    b_size = 31
    
    A = list(range(a_size))
    B = list(range(a_size, n))
    
    edges = set()
    
    # 1. Add all edges between A and B
    for a in A:
        for b in B:
            edges.add((a, b))
            
    needed_in_B = target_edges - (a_size * b_size) # 122
    
    def try_construct_B(max_deg):
        B_edges = []
        degree = {v: 0 for v in B}
        adj = {v: set() for v in B}
        
        # Generate pairs and shuffle
        pairs = list(itertools.combinations(B, 2))
        random.shuffle(pairs)
        
        for u, v in pairs:
            if len(B_edges) == needed_in_B:
                break
            
            # Check triangle constraint (common neighbor in B)
            if adj[u] & adj[v]:
                continue
                
            # Check degree constraint
            if degree[u] < max_deg and degree[v] < max_deg:
                B_edges.append((u, v))
                degree[u] += 1
                degree[v] += 1
                adj[u].add(v)
                adj[v].add(u)
                
        if len(B_edges) == needed_in_B:
            return B_edges
        return None

    B_edges = None
    # Try to find a solution with max degree 9 (allows triangle-free)
    # max_deg 8 is too tight for 122 edges in 31 vertices (max 124, but greedy might fail)
    # max_deg 9 is safe (max 139)
    attempts = 0
    while time.time() < deadline - 10.0 and attempts < 200:
        B_edges = try_construct_B(9)
        if B_edges:
            break
        attempts += 1
        
    # Fallback to 10 if 9 fails (should not happen)
    if not B_edges:
        B_edges = try_construct_B(10)
        
    if not B_edges:
        # Last resort: random edges, likely high score
        pairs = list(itertools.combinations(B, 2))
        random.shuffle(pairs)
        B_edges = pairs[:needed_in_B]

    for u, v in B_edges:
        edges.add((u, v))
        
    result = [[u, v] for u, v in edges]
    
    return result
