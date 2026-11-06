"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    def instance(n, d, seed):
        from random import Random
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
    
    def strong_color(edges, adj):
        m = len(edges)
        if m == 0:
            return []
        
        # Build adjacency in L^2(G): two edges are adjacent if they share a vertex or are connected by an edge
        # edge_i = (u,v), edge_j = (x,y)
        # Adjacent if: {u,v} & {x,y} != empty OR (u,x) in adj_set OR (u,y) in adj_set OR (v,x) in adj_set OR (v,y) in adj_set
        adj_set = set()
        for u, nbrs in adj.items():
            for v in nbrs:
                adj_set.add((u, v))
                adj_set.add((v, u))
        
        # Build conflict graph on edges
        # For efficiency, use bitmasks or just sets
        edge_adj = [set() for _ in range(m)]
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                # Check if they share a vertex
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    edge_adj[i].add(j)
                    edge_adj[j].add(i)
                    continue
                # Check if connected by an edge
                if (u1, u2) in adj_set or (u1, v2) in adj_set or (v1, u2) in adj_set or (v1, v2) in adj_set:
                    edge_adj[i].add(j)
                    edge_adj[j].add(i)
        
        # DSATUR greedy coloring
        color = [-1] * m
        color_count = [0] * m  # number of colors used by neighbors
        # We'll use a priority queue approach: pick uncolored vertex with max saturation, then max degree
        
        import heapq
        
        # For each vertex, track which colors are used by its colored neighbors
        neighbor_colors = [set() for _ in range(m)]
        
        uncolored = set(range(m))
        best_order = None
        best_num_colors = m + 1
        
        # Try a few different orderings via DSATUR
        for attempt in range(3):
            color = [-1] * m
            neighbor_colors = [set() for _ in range(m)]
            uncolored = list(range(m))
            num_colors = 0
            
            while uncolored:
                # Pick vertex with max saturation (number of distinct colors in neighbors), break ties by degree
                best_v = None
                best_sat = -1
                best_deg = -1
                for v in uncolored:
                    sat = len(neighbor_colors[v])
                    deg = len(edge_adj[v])
                    if sat > best_sat or (sat == best_sat and deg > best_deg):
                        best_sat = sat
                        best_deg = deg
                        best_v = v
                
                # Assign smallest available color
                used = neighbor_colors[best_v]
                c = 0
                while c in used:
                    c += 1
                color[best_v] = c
                if c + 1 > num_colors:
                    num_colors = c + 1
                
                # Update neighbors
                for nb in edge_adj[best_v]:
                    if nb in uncolored:
                        neighbor_colors[nb].add(c)
                
                uncolored.remove(best_v)
            
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_order = color[:]
        
        return best_order
    
    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        colors = strong_color(edges, adj)
        out.append(colors)
    return out
# EVOLVE-BLOCK-END
