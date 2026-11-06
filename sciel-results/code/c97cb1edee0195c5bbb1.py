import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        # Ensure we try enough times to get a good graph
        # The original used 4*n*d attempts.
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_graph(edges, adj):
        # Map edge to index
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        n_edges = len(edges)
        # Conflict adjacency
        # Two edges (u,v) and (x,y) conflict if:
        # 1. They share a vertex (intersection is not empty)
        # 2. They are "close": one endpoint of one is connected to one endpoint of the other
        # i.e., (u,x) in adj or (u,y) in adj or (v,x) in adj or (v,y) in adj
        # Note: If they share a vertex, they are already conflicting.
        
        conflict_adj = [set() for _ in range(n_edges)]
        
        for i, (u, v) in enumerate(edges):
            # Check against all other edges? O(M^2) is fine for M=240 (57600 ops)
            # But we can optimize using adjacency.
            # For edge (u,v), conflicting edges are:
            # - Edges incident to u or v (share vertex)
            # - Edges (x,y) such that x is neighbor of u or v, or y is neighbor of u or v.
            # This is getting complex to iterate efficiently without pre-computation.
            # Given M is small, O(M^2) with simple checks is fast enough in Python if optimized.
            pass
            
        # Simpler O(M^2) approach
        for i in range(n_edges):
            u1, v1 = edges[i]
            for j in range(i + 1, n_edges):
                u2, v2 = edges[j]
                # Check 1: Share vertex
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict_adj[i].add(j)
                    conflict_adj[j].add(i)
                    continue
                # Check 2: Connected by an edge
                # (u1, u2) in adj? (u1, v2) in adj? (v1, u2) in adj? (v1, v2) in adj?
                if (u1 in adj.get(u2, set()) or 
                    u1 in adj.get(v2, set()) or 
                    v1 in adj.get(u2, set()) or 
                    v1 in adj.get(v2, set())):
                    conflict_adj[i].add(j)
                    conflict_adj[j].add(i)
                    
        return conflict_adj

    def dsatur_coloring(conflict_adj):
        n = len(conflict_adj)
        colors = [-1] * n
        # sat[i] = set of colors used by neighbors of i
        sat = [set() for _ in range(n)]
        # uncolored set
        uncolored = set(range(n))
        
        # Precompute degrees for tie-breaking
        degrees = [len(conflict_adj[i]) for i in range(n)]
        
        while uncolored:
            # Select vertex with max saturation, tie-break by degree
            # For small N, linear scan is fine
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in uncolored:
                s = len(sat[v])
                if s > best_sat or (s == best_sat and degrees[v] > best_deg):
                    best_sat = s
                    best_deg = degrees[v]
                    best_v = v
            
            # Assign smallest available color
            c = 0
            while c in sat[best_v]:
                c += 1
            
            colors[best_v] = c
            uncolored.remove(best_v)
            
            # Update neighbors
            for nb in conflict_adj[best_v]:
                if nb in uncolored:
                    sat[nb].add(c)
                    
        return colors

    def optimize_coloring(colors, conflict_adj):
        # Try to reduce the number of colors
        # colors is a list of ints
        n = len(colors)
        if n == 0: return colors
        
        max_c = max(colors)
        # Map color to list of vertices
        color_to_verts = defaultdict(list)
        for i, c in enumerate(colors):
            color_to_verts[c].append(i)
            
        # We want to reduce max_c
        # Strategy: Try to recolor all vertices of color max_c to lower colors.
        # If successful, max_c decreases.
        
        # To make it more robust, we can iterate this process.
        # Also, we can try to swap colors if direct recoloring fails.
        
        # Simple local search:
        # Repeat until no improvement or time limit
        improved = True
        while improved:
            improved = False
            if not color_to_verts[max_c]:
                max_c -= 1
                if max_c < 0: break
                continue
                
            # Try to recolor each vertex in color_to_verts[max_c]
            # We must be careful: if we recolor v1 to c1, it might block v2.
            # So we should try to find a consistent assignment for all vertices of max_c.
            # This is a subgraph coloring problem.
            # Since the subgraph of max_c vertices is an independent set in the conflict graph?
            # No, they are all color max_c, so they are NOT adjacent in the conflict graph?
            # Wait, if they are adjacent in conflict graph, they cannot share the same color.
            # So vertices with the same color form an independent set in the conflict graph.
            # So we can recolor them one by one, but we must ensure the new color is valid.
            
            # Let's try a greedy recoloring of the max_c vertices.
            # Order them by degree in the conflict graph (highest first) to reduce conflicts early.
            verts_to_recolor = sorted(color_to_verts[max_c], key=lambda x: len(conflict_adj[x]), reverse=True)
            
            # We need to track which colors are "used" by neighbors for each vertex.
            # But if we change a color, it affects neighbors.
            # This is tricky. 
            # Alternative: Just try to find ANY vertex that can be recolored to a lower color.
            # If we can, do it. This is a "first-fit improvement".
            
            recolored_any = False
            # Shuffle to avoid bias? Or just iterate.
            for v in verts_to_recolor:
                # Check if v can be recolored to any c in [0, max_c-1]
                # The forbidden colors for v are those used by its neighbors.
                # Note: Neighbors might include other vertices currently colored max_c.
                # If we recolor v, we free up max_c for v, but we use a new color c.
                # The neighbors of v that are also max_c are not adjacent to v (independent set).
                # So their color max_c is not in the forbidden set for v (since they are not neighbors).
                # So we just need to check neighbors with color < max_c.
                
                forbidden = set()
                for nb in conflict_adj[v]:
                    if colors[nb] != max_c: # Neighbors with max_c are not adjacent to v? 
                        # Wait, if nb is neighbor of v, they conflict. 
                        # If colors[nb] == max_c and colors[v] == max_c, that's a conflict!
                        # But colors[v] is currently max_c. 
                        # In a valid coloring, no two adjacent vertices have the same color.
                        # So if nb is neighbor of v, colors[nb] != colors[v].
                        # So colors[nb] cannot be max_c if colors[v] is max_c.
                        # Thus, all neighbors of v have colors != max_c.
                        forbidden.add(colors[nb])
                
                # Find smallest c < max_c not in forbidden
                c_new = 0
                while c_new in forbidden and c_new < max_c:
                    c_new += 1
                    
                if c_new < max_c:
                    # Recolor
                    colors[v] = c_new
                    color_to_verts[max_c].remove(v)
                    color_to_verts[c_new].append(v)
                    recolored_any = True
                    # If we recolored, the set of forbidden colors for other max_c vertices might change?
                    # Only if they are neighbors of v. But they are not (independent set).
                    # So we can continue checking other vertices.
                    
            if not color_to_verts[max_c]:
                max_c -= 1
                improved = True
            elif recolored_any:
                improved = True
            else:
                # Stuck. Try a random restart or different heuristic?
                # For now, break.
                break
                
        return colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline - 10:
            # Not enough time, use naive or break? 
            # We must return something valid. 
            # If we haven't processed this instance, we should process it quickly.
            # Let's just process it with DSATUR only.
            pass

        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        if not edges:
            results.append([])
            continue
            
        conflict_adj = build_conflict_graph(edges, adj)
        colors = dsatur_coloring(conflict_adj)
        
        # Optimize
        colors = optimize_coloring(colors, conflict_adj)
        
        results.append(colors)
        
    return results
