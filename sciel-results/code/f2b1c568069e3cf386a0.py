import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    target_edges = 401
    
    # Partition sizes
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
    
    # We want to construct G_B on B with 122 edges, triangle-free, max degree 8.
    # If successful, score is 8.
    # If not, we fall back to max degree 9, score 9.
    
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
            
            # Check triangle constraint
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
    # Try to find a solution with max degree 8
    # We will try several random restarts
    attempts = 0
    while time.time() < deadline - 10.0 and attempts < 50:
        B_edges = try_construct_B(8)
        if B_edges:
            break
        attempts += 1
        
    # If we failed with max degree 8, try max degree 9
    if not B_edges:
        B_edges = try_construct_B(9)
        
    # If we still failed (unlikely), just pick any 122 edges
    if not B_edges:
        pairs = list(itertools.combinations(B, 2))
        random.shuffle(pairs)
        B_edges = pairs[:needed_in_B]

    # Add B edges to main set
    for u, v in B_edges:
        edges.add((u, v))
        
    # Format output
    result = [[u, v] for u, v in edges]
    
    return result
