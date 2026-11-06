"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    from collections import defaultdict

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
        """
        colors = [-1] * num_vertices
        sat = [set() for _ in range(num_vertices)]
        degrees = [len(conflict_adj[i]) for i in range(num_vertices)]
        uncolored = set(range(num_vertices))
        
        while uncolored:
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
            
            used_colors = sat[best_v]
            c = 0
            while c in used_colors:
                c += 1
            
            colors[best_v] = c
            uncolored.remove(best_v)
            
            for u in conflict_adj[best_v]:
                if u in uncolored:
                    sat[u].add(c)

        return colors

    def greedy_coloring(conflict_adj, num_vertices, order):
        """
        Greedy coloring following a specific order.
        """
        colors = [-1] * num_vertices
        # used_colors_by_neighbor is not needed if we check neighbors directly
        # But checking neighbors is O(degree).
        # To speed up, we can maintain a set of forbidden colors for each vertex.
        forbidden = [set() for _ in range(num_vertices)]
        
        for v in order:
            c = 0
            while c in forbidden[v]:
                c += 1
            colors[v] = c
            for u in conflict_adj[v]:
                forbidden[u].add(c)
        
        return colors

    def local_search_improve(colors, conflict_adj, num_vertices):
        """
        Try to reduce colors by recoloring vertices.
        """
        # Map color -> list of vertices
        color_classes = defaultdict(list)
        for v, c in enumerate(colors):
            color_classes[c].append(v)
        
        max_color = max(colors) if colors else 0
        
        # Try to remove the highest color
        # We iterate from high to low, trying to recolor vertices of that color to lower colors
        for c in range(max_color, -1, -1):
            if not color_classes[c]:
                continue
            
            # Try to recolor each vertex in this class
            # If we can recolor all vertices in this class to other colors, we remove this color
            # This is a heuristic: we try one by one.
            changed = False
            for v in list(color_classes[c]):
                # Find a new color
                new_c = -1
                # Check colors 0 to max_color
                for nc in range(max_color):
                    if nc == c:
                        continue
                    # Check if nc is forbidden
                    conflict = False
                    for u in conflict_adj[v]:
                        if colors[u] == nc:
                            conflict = True
                            break
                    if not conflict:
                        new_c = nc
                        break
                
                if new_c != -1:
                    colors[v] = new_c
                    color_classes[c].remove(v)
                    color_classes[new_c].append(v)
                    changed = True
            
            # If we couldn't recolor any, we keep the color
            # Note: This simple local search might not be optimal but can help.
            # To be safer, we only accept if we actually reduced the max color used or kept it same.
            # Since we are iterating down, if we succeed in recoloring all of color c, 
            # the max color might drop.
            
        return colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        n = len(edges)
        
        # Build original graph adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Build conflict graph
        conflict_adj = [set() for _ in range(n)]
        
        for i in range(n):
            u1, v1 = edges[i]
            for j in range(i + 1, n):
                u2, v2 = edges[j]
                
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict_adj[i].add(j)
                    conflict_adj[j].add(i)
                else:
                    if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                        conflict_adj[i].add(j)
                        conflict_adj[j].add(i)

        # Strategy 1: DSATUR
        colors_dsatur = dsatur_coloring(conflict_adj, n)
        num_colors_dsatur = max(colors_dsatur) + 1 if n > 0 else 0
        
        # Strategy 2: Greedy with degree ordering
        degrees = [len(conflict_adj[i]) for i in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        colors_greedy = greedy_coloring(conflict_adj, n, order)
        num_colors_greedy = max(colors_greedy) + 1 if n > 0 else 0
        
        # Choose the better one
        if num_colors_dsatur <= num_colors_greedy:
            best_colors = colors_dsatur
        else:
            best_colors = colors_greedy
            
        # Apply local search improvement
        best_colors = local_search_improve(best_colors, conflict_adj, n)
        
        out.append(best_colors)

    return out
# EVOLVE-BLOCK-END
