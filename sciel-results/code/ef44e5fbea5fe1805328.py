"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def dsatur_coloring(conflict_adj, num_vertices):
        """
        Colors a graph using the DSATUR heuristic.
        conflict_adj: list of lists of neighbors.
        num_vertices: number of vertices.
        Returns: list of colors for each vertex.
        """
        if num_vertices == 0:
            return []
        
        colors = [-1] * num_vertices
        # sat[i] is the set of colors used by neighbors of i
        sat = [set() for _ in range(num_vertices)]
        
        # Precompute degrees for tie-breaking
        degrees = [len(conflict_adj[i]) for i in range(num_vertices)]
        
        uncolored = set(range(num_vertices))
        
        while uncolored:
            # Find the vertex with the highest saturation degree, then highest degree
            best_v = -1
            best_sat = -1
            best_deg = -1
            
            for v in uncolored:
                s = len(sat[v])
                d = degrees[v]
                if s > best_sat or (s == best_sat and d > best_deg):
                    best_sat = s
                    best_deg = d
                    best_v = v
            
            # Assign the smallest available color
            c = 0
            while c in sat[best_v]:
                c += 1
            
            colors[best_v] = c
            
            # Update saturation of neighbors
            for u in conflict_adj[best_v]:
                if u in uncolored:
                    sat[u].add(c)
            
            uncolored.remove(best_v)
            
        return colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        n_edges = len(edges)
        
        # Build original graph adjacency
        adj = {u: set() for u in range(100)} 
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Build conflict graph
        # Vertices are 0..n_edges-1
        conflict_adj = [set() for _ in range(n_edges)]
        
        for i in range(n_edges):
            u1, v1 = edges[i]
            for j in range(i + 1, n_edges):
                u2, v2 = edges[j]
                # Check if they share a vertex
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict_adj[i].add(j)
                    conflict_adj[j].add(i)
                    continue
                
                # Check if joined by an edge
                if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                    conflict_adj[i].add(j)
                    conflict_adj[j].add(i)
                    
        # Convert sets to lists for DSATUR
        conflict_adj_list = [list(s) for s in conflict_adj]
        
        colors = dsatur_coloring(conflict_adj_list, n_edges)
        out.append(colors)
        
    return out
# EVOLVE-BLOCK-END
