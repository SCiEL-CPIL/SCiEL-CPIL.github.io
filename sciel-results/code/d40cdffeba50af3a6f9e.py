<<<<<<< SEARCH
    # Initialize with a few different strategies
    # Strategy 1: Start with Grötzsch graph
    # Vertices 0-4: C5
    # Vertices 5-9: copies
    # Vertex 10: universal to 5-9? No, Mycielski construction:
    # V' = V U V' U {w}
    # Edges: (v, v') if v in V and v' is copy?
    # Standard Mycielski:
    # Given G=(V,E), M(G) has V U V' U {w}.
    # Edges in G.
    # For each edge (u,v) in G, add (u, v') and (v, u').
    # Connect w to all v' in V'.
    
    def make_grotzsch():
        adj = [0] * n
        # C5: 0-1-2-3-4-0
        cycle = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        for u, v in cycle:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Copies 5-9 for 0-4
        for u, v in cycle:
            # (u, v') => (u, 5+v)
            adj[u] |= (1 << (5+v))
            adj[5+v] |= (1 << u)
            # (v, u') => (v, 5+u)
            adj[v] |= (1 << (5+u))
            adj[5+u] |= (1 << v)
            
        # w=10 connected to 5-9
        for i in range(5):
            adj[10] |= (1 << (5+i))
            adj[5+i] |= (1 << 10)
            
        return adj

    # Strategy 2: Random triangle-free graph
    def make_random_tf():
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    if is_triangle_free_adj(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        return adj
=======
    def make_grotzsch():
        adj = [0] * n
        # C5: 0-1-2-3-4-0
        cycle = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        for u, v in cycle:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Copies 5-9 for 0-4
        for u, v in cycle:
            adj[u] |= (1 << (5+v))
            adj[5+v] |= (1 << u)
            adj[v] |= (1 << (5+u))
            adj[5+u] |= (1 << v)
            
        # w=10 connected to 5-9
        for i in range(5):
            adj[10] |= (1 << (5+i))
            adj[5+i] |= (1 << 10)
            
        return adj

    def make_random_tf(density=0.5):
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < density:
                    if is_triangle_free_adj(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        return adj

    def make_bipartite_complete():
        """Create a complete bipartite graph K_{a,b} which is triangle-free and 2-colorable,
        then try to add edges to make it not 3-colorable"""
        adj = [0] * n
        a = n // 2
        b = n - a
        # Partition: 0..a-1 and a..n-1
        for u in range(a):
            for v in range(a, n):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
>>>>>>> REPLACE

