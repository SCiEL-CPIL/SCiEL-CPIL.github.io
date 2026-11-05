import random
import time

def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    best_edges = None
    best_score = float('inf')
    
    n = 40
    m = 401
    
    # Generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for seed in range(10000):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        edges = random.sample(all_edges, m)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Compute max book size
        max_book = 0
        for u, v in edges:
            common = len(adj[u] & adj[v])
            if common > max_book:
                max_book = common
                if max_book >= best_score:
                    break
        
        if max_book < best_score:
            best_score = max_book
            best_edges = edges
    
    # Convert to list of lists
    if best_edges is None:
        # Fallback: K_{20,20} + one edge
        best_edges = [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    
    return [[u, v] for u, v in best_edges]
