def construct():
    from random import Random
    import heapq
    import time
    
    deadline = time.time() + 140  # safety margin before 150s limit
    
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
    
    def build_conflict_graph(edges, n):
        """Build L(G)^2 conflict graph. Two edges conflict if they share a vertex
        or are connected by an edge in G."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        m = len(edges)
        # Map edge to index
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        # For each edge, find all conflicting edges
        conflict = [set() for _ in range(m)]
        
        # For each edge (u,v), conflicting edges are:
        # 1. All edges sharing u or v
        # 2. All edges (x,y) where x in {u,v}'s neighbors or x in {u,v}, and y in {u,v}'s neighbors or y in {u,v}
        #    More precisely: edges (x,y) where {x,y} intersects N[{u,v}] where N[{u,v}] is closed neighbourhood
        #    But we need to exclude the edge (u,v) itself and edges that share a vertex (already counted)
        
        # Let's think more carefully. Edges e1=(u,v) and e2=(x,y) conflict if:
        # - They share a vertex: {u,v} ∩ {x,y} ≠ ∅
        # - OR there's an edge connecting an endpoint of e1 to an endpoint of e2:
        #   (u,x) in E or (u,y) in E or (v,x) in E or (v,y) in E
        #   Note: this includes the case where x or y equals u or v, but those are already covered by shared vertex
        
        # So the conflict set for edge (u,v) is all edges (x,y) such that:
        # {x,y} ∩ ({u,v} ∪ N(u) ∪ N(v)) ≠ ∅
        # where N(u) is the open neighbourhood of u
        
        # Actually, let me re-derive. Two edges e1=(u,v) and e2=(x,y) are in the same induced matching
        # iff:
        # - u,v,x,y are all distinct (no shared vertex)
        # - No edge connects {u,v} to {x,y} in G
        # The second condition means: for all a in {u,v}, b in {x,y}, (a,b) not in E
        
        # So e1 and e2 conflict iff NOT (all four distinct AND no connecting edge)
        # i.e., conflict iff shared vertex OR connecting edge exists
        
        # For edge (u,v), the set of vertices that "touch" it is {u,v} ∪ N(u) ∪ N(v)
        # Any edge (x,y) where x or y is in this set will conflict with (u,v)
        # Wait, that's not quite right. If x is in N(u), then (u,x) is an edge, so (u,v) and (x,y) have a connecting edge. Yes.
        # If x = u, they share vertex. Yes.
        # So: e1=(u,v) conflicts with e2=(x,y) iff x in S or y in S, where S = {u,v} ∪ N(u) ∪ N(v)
        
        for i, (u, v) in enumerate(edges):
            S = {u, v} | adj[u] | adj[v]
            for j, (x, y) in enumerate(edges):
                if i == j:
                    continue
                if x in S or y in S:
                    conflict[i].add(j)
                    conflict[j].add(i)
        
        return conflict
    
    def dsatur_coloring(conflict):
        """DSATUR heuristic for graph colouring."""
        m = len(conflict)
        color = [-1] * m
        # Track for each colour, which vertices use it
        color_classes = []
        # Saturation degree: number of different colours in neighbourhood
        sat_deg = [0] * m
        # For each vertex, set of colours used by its neighbours
        neighbour_colors = [set() for _ in range(m)]
        
        # Priority queue: (-sat_deg, -degree, vertex)
        # We'll use a simple approach: pick uncoloured vertex with max saturation, then max degree
        
        uncoloured = list(range(m))
        degrees = [len(conflict[i]) for i in range(m)]
        
        while uncoloured:
            # Find vertex with max saturation degree, break ties by degree
            best = None
            best_sat = -1
            best_deg = -1
            for v in uncoloured:
                s = len(neighbour_colors[v])
                if s > best_sat or (s == best_sat and degrees[v] > best_deg):
                    best = v
                    best_sat = s
                    best_deg = degrees[v]
            
            uncoloured.remove(best)
            
            # Find smallest available colour
            used = neighbour_colors[best]
            c = 0
            while c in used:
                c += 1
            
            color[best] = c
            if c >= len(color_classes):
                color_classes.append([])
            color_classes[c].append(best)
            
            # Update neighbours
            for nb in conflict[best]:
                if color[nb] == -1:
                    before = len(neighbour_colors[nb])
                    neighbour_colors[nb].add(c)
                    after = len(neighbour_colors[nb])
                    # sat_deg updated implicitly
        
        return color
    
    def local_search_improve(color, conflict, max_iters=5000):
        """Try to reduce colour count via local search."""
        m = len(conflict)
        color = color[:]
        
        # Count colours used
        def num_colours(col):
            return max(col) + 1 if col else 0
        
        best_color = color[:]
        best_nc = num_colours(best_color)
        
        for _ in range(max_iters):
            improved = False
            nc = num_colours(color)
            
            # Try to move each vertex to a lower colour if possible
            for v in range(m):
                current_c = color[v]
                # Try colours 0 to current_c-1
                for c in range(current_c):
                    # Check if v can take colour c
                    ok = True
                    for nb in conflict[v]:
                        if color[nb] == c:
                            ok = False
                            break
                    if ok:
                        color[v] = c
                        improved = True
                        break
            
            # Try to eliminate the highest colour
            if nc > 0:
                max_c = nc - 1
                # Find vertices with colour max_c
                vertices_with_max = [v for v in range(m) if color[v] == max_c]
                for v in vertices_with_max:
                    # Try to move v to any other colour
                    for c in range(max_c):
                        ok = True
                        for nb in conflict[v]:
                            if color[nb] == c:
                                ok = False
                                break
                        if ok:
                            color[v] = c
                            improved = True
                            break
            
            if not improved:
                break
            
            nc_new = num_colours(color)
            if nc_new < best_nc:
                best_nc = nc_new
                best_color = color[:]
        
        return best_color
    
    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline:
            # Return best effort
            break
        edges = instance(*spec)
        n = spec[0]
        conflict = build_conflict_graph(edges, n)
        color = dsatur_coloring(conflict)
        color = local_search_improve(color, conflict)
        out.append(color)
    
    # If we ran out of time, fill remaining with simple greedy
    while len(out) < 3:
        spec = ((40, 4, 101), (60, 5, 102), (80, 6, 103))[len(out)]
        edges = instance(*spec)
        n = spec[0]
        conflict = build_conflict_graph(edges, n)
        # Simple greedy
        m = len(edges)
        color = [-1] * m
        for i in range(m):
            used = set()
            for nb in conflict[i]:
                if color[nb] != -1:
                    used.add(color[nb])
            c = 0
            while c in used:
                c += 1
            color[i] = c
        out.append(color)
    
    return out
