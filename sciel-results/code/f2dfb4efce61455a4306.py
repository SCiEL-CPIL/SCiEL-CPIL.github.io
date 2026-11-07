"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 90.0 # 90s wall-clock limit, use 89s to be safe
    
    n = 16
    # The maximum number of edges in a triangle-free graph on n vertices is floor(n^2/4).
    # Such graphs (specifically complete bipartite) have diameter 2 (if both partitions non-empty and size > 1? 
    # K_{1,15} has diameter 2? d(u,v) for u,v in B is 2. d(u, v) for u in A, v in B is 1. Yes.)
    # And every edge is essential because the graph is triangle-free.
    # For n=16, K_{8,8} has 64 edges.
    
    # Check time
    if time.time() - start_time > deadline:
        # Fallback to a smaller valid graph if we somehow took too long (unlikely)
        return []
        
    set_a = list(range(8))
    set_b = list(range(8, 16))
    
    edges = []
    for u in set_a:
        for v in set_b:
            edges.append([u, v])
            
    # Final check
    if time.time() - start_time > deadline:
        return edges
        
    return edges
# EVOLVE-BLOCK-END
