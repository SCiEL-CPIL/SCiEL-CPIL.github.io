"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = 20
    # Use a known dense K2,3-free construction or greedy approach
    # For n=20, a good starting point is to use a 5-regular or 6-regular graph
    # that is K2,3-free, then try to add edges greedily.
    
    # Start with a 5-regular graph: 5 copies of K5? No, K5 has K2,3.
    # K2,3-free means no two vertices share 3 neighbors.
    # In K5, any two vertices share 3 neighbors. So K5 is not K2,3-free.
    # Max degree in K2,3-free graph: if two vertices have degree d, they can share at most 2 neighbors.
    
    # Let's use a greedy algorithm to build a dense K2,3-free graph.
    # Start with empty graph, add edges in order of increasing "constraint".
    
    # Better: use a known construction. For n=20, the extremal number ex(20, K2,3) is known.
    # According to literature, ex(n, K2,3) ~ n^(3/2)/2 for large n.
    # For n=20, n^(3/2)/2 = 20*sqrt(20)/2 = 10*4.47 = 44.7. So around 44-45 edges.
    
    # Let's try to construct a graph with ~45 edges that is K2,3-free.
    # Use a greedy algorithm with backtracking if time permits.
    
    # Simple greedy: add all possible edges, then remove edges that create K2,3.
    # Start with complete graph, remove edges to eliminate K2,3.
    
    import numpy as np
    
    # Adjacency matrix
    adj = np.ones((n, n), dtype=bool)
    np.fill_diagonal(adj, False)
    
    def is_k23_free(adj):
        """Check if graph is K2,3-free."""
        for u in range(n):
            for v in range(u + 1, n):
                # Count common neighbors
                common = np.sum(adj[u] & adj[v])
                if common >= 3:
                    return False
        return True
    
    def common_neighbors(adj, u, v):
        return np.where(adj[u] & adj[v])[0]
    
    # Greedy removal: while there's a K2,3, remove an edge from it.
    # A K2,3 consists of two vertices u,v with at least 3 common neighbors.
    # To break it, we can remove an edge from u to a common neighbor, or v to a common neighbor.
    
    max_iter = 5000
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Find a K2,3
        found = False
        for u in range(n):
            for v in range(u + 1, n):
                common = common_neighbors(adj, u, v)
                if len(common) >= 3:
                    found = True
                    # Remove an edge to break this K2,3
                    # Choose a random common neighbor and remove edge (u, common[0]) or (v, common[0])
                    w = common[0]
                    # Remove edge (u, w)
                    adj[u, w] = False
                    adj[w, u] = False
                    break
            if found:
                break
        
        if not found:
            break
    
    # Extract edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                edges.append([u, v])
    
    # Try to add more edges greedily
    for _ in range(100):
        if time.time() > deadline:
            break
        
        # Try to add each missing edge if it doesn't create K2,3
        added = False
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u, v]:
                    # Check if adding (u,v) creates K2,3
                    # Adding (u,v) increases common neighbors of u with all neighbors of v, and v with all neighbors of u
                    # We need to check if any pair of vertices will have >= 3 common neighbors after adding (u,v)
                    
                    # The new edges affect:
                    # 1. Pair (u, x) for x in neighbors of v: common neighbors of u and x increase by 1 if v is common
                    # 2. Pair (v, x) for x in neighbors of u: common neighbors of v and x increase by 1 if u is common
                    # 3. Pair (u, v): they now have common neighbors = neighbors of u intersect neighbors of v
                    
                    # Simpler: just add the edge and check if graph is still K2,3-free
                    adj[u, v] = True
                    adj[v, u] = True
                    
                    if is_k23_free(adj):
                        added = True
                        break
                    else:
                        adj[u, v] = False
                        adj[v, u] = False
            
            if added:
                break
        
        if not added:
            break
    
    # Extract edges again
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
