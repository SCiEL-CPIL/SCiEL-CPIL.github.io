import time
import random

def construct():
    start_time = time.time()
    # Use 110s to be safe within 120s limit
    deadline = start_time + 110.0
    
    n = 20
    vertices = list(range(n))
    # List of all possible edges (u, v) with u < v
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_edge_count = 0
    best_result = []
    
    # We can also try to improve by removing edges from a complete graph?
    # No, building up is safer for the C4-free constraint.
    
    # To increase diversity, we can also try different random seeds or just loop.
    
    while time.time() < deadline:
        random.shuffle(all_edges)
        
        # Adjacency list using sets for fast intersection
        adj = [set() for _ in range(n)]
        current_edges = []
        
        for u, v in all_edges:
            # Check if adding (u, v) creates a C4
            # A C4 is formed if there are at least 2 common neighbors
            
            # Optimization: Iterate over the smaller neighbor set
            if len(adj[u]) < len(adj[v]):
                smaller = adj[u]
                larger = adj[v]
            else:
                smaller = adj[v]
                larger = adj[u]
                
            common_count = 0
            for w in smaller:
                if w in larger:
                    common_count += 1
                    if common_count >= 2:
                        break
                        
            if common_count < 2:
                # Safe to add
                adj[u].add(v)
                adj[v].add(u)
                
                # Assign a color. 
                # Since the graph is C4-free, any color assignment is valid 
                # with respect to the "rainbow 4-cycle" constraint.
                # We use a simple deterministic color based on edge index 
                # to ensure variety, though not strictly necessary.
                c = len(current_edges) % 20
                current_edges.append([u, v, c])
                
        if len(current_edges) > best_edge_count:
            best_edge_count = len(current_edges)
            best_result = current_edges
            
            # If we find a very dense graph, we might stop early?
            # Theoretical max for C4-free on n=20 is ~90.
            # If we hit 100, something is wrong with the logic (it shouldn't happen).
            # If we hit the theoretical max, we can stop.
            # But we don't know the exact max.
            
    return best_result
