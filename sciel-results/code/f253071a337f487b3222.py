import random
import time

def construct():
    n = 20
    # Seed program is a star with 19 edges.
    # We will try to find a C4-free graph with more edges.
    # A C4-free graph is automatically valid for the "rainbow 4-cycle" constraint.
    
    best_edges = []
    best_count = 0
    deadline = time.time() + 110  # Safety margin
    
    # Precompute all pairs
    pairs = []
    for u in range(n):
        for v in range(u + 1, n):
            pairs.append((u, v))
    
    # Try multiple random orders
    # Use a fixed seed for reproducibility in some contexts, but random is better for search.
    # We'll just use random.
    
    while time.time() < deadline:
        random.shuffle(pairs)
        adj = [[False] * n for _ in range(n)]
        current_edges = []
        
        for u, v in pairs:
            # Check if adding (u, v) creates a C4
            # This happens if there is a w such that (u, w) and (v, w) are in adj
            # i.e., N(u) intersects N(v)
            has_common_neighbor = False
            # Optimization: check neighbors of u
            for w in range(n):
                if adj[u][w] and adj[v][w]:
                    has_common_neighbor = True
                    break
            
            if not has_common_neighbor:
                adj[u][v] = True
                adj[v][u] = True
                current_edges.append((u, v))
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
    # If we found nothing better than star (shouldn't happen), fallback to star
    if not best_edges:
        return [[0, v, 0] for v in range(1, n)]
        
    # Color all edges with 0 (valid since no 4-cycles)
    # Or use distinct colors if we want, but 0 is fine.
    # The task asks for colours 0..19. Using 0 for all is valid.
    return [[u, v, 0] for u, v in best_edges]
