import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s/150s limit
    
    n = 40
    m = 401
    
    best_edges = None
    best_score = float('inf')
    
    # Precompute all possible edges
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((i, j))
            
    # Baseline: K20,20 + 1 edge
    # This gives score 20. We hope to do better.
    # Actually, let's just start with random sampling.
    
    while time.time() < deadline:
        # Generate a random set of m edges
        random.shuffle(all_edges)
        edges = all_edges[:m]
        
        # Build adjacency sets for fast common neighbor calculation
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v) if u < v else (v, u))
        
        # Calculate max book size
        max_book = 0
        for u, v in edges:
            # Number of common neighbors
            common = len(adj[u] & adj[v])
            if common > max_book:
                max_book = common
                # Early termination if we already exceed best known
                if max_book >= best_score:
                    break
        
        if max_book < best_score:
            best_score = max_book
            best_edges = edges.copy()
            
            # If we hit the theoretical lower bound, we can stop
            if best_score <= 7:
                break

    if best_edges is None:
        # Fallback to baseline if something went wrong
        best_edges = [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    # Format output as list of lists [u, v]
    return [[u, v] for u, v in best_edges]

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
