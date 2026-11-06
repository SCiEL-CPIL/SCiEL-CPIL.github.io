import time
import random

def construct():
    n = 16
    deadline = time.time() + 110  # 110 seconds to be safe
    
    # Grötzsch graph edges (11 vertices, 20 edges)
    # Vertices 0-4: C5
    # Vertices 5-9: Mycielski layer
    # Vertex 10: Mycielski top
    seed_edges = []
    for i in range(5):
        seed_edges.append((i, (i + 1) % 5))
    for u, w in [(i, (i + 1) % 5) for i in range(5)]:
        seed_edges.append((u, 5 + w))
        seed_edges.append((w, 5 + u))
    for u in range(5):
        seed_edges.append((5 + u, 10))
        
    # Convert to bitmask adjacency
    adj = [0] * n
    for u, v in seed_edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        
    # List of all potential edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_count = len(seed_edges)
    best_edges = seed_edges[:]
    
    # Helper to check if adding edge (u, v) is valid (no triangle)
    def is_valid_add(u, v, adj):
        if (adj[u] >> v) & 1:
            return False # Already exists
        # Check triangle
        if (adj[u] & adj[v]):
            return False
        return True

    # Local search with random restarts
    # Since the property is monotone, we just want to find a maximal triangle-free supergraph
    # with the maximum number of edges.
    
    for restart in range(2000): # Enough restarts to finish quickly
        if time.time() > deadline:
            break
            
        current_adj = adj[:]
        current_edges = set(seed_edges)
        
        # Create a list of candidate edges
        candidates = []
        for u, v in all_edges:
            if (u, v) not in current_edges:
                if is_valid_add(u, v, current_adj):
                    candidates.append((u, v))
        
        # Randomly add valid edges until no more can be added
        random.shuffle(candidates)
        for u, v in candidates:
            if is_valid_add(u, v, current_adj):
                current_adj[u] |= 1 << v
                current_adj[v] |= 1 << u
                current_edges.add((u, v))
        
        count = len(current_edges)
        if count > best_count:
            best_count = count
            best_edges = list(current_edges)
            
    # Convert best_edges to list of lists [u, v]
    return [[u, v] for u, v in best_edges]
