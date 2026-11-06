"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys

def construct():
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        # Ensure we try enough times to get a connected-ish graph
        for _ in range(10 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(edges, adj):
        m = len(edges)
        if m == 0:
            return []
        
        # Build conflict graph
        # edge i conflicts with edge j if they share a vertex or are connected by an edge
        conflict = [set() for _ in range(m)]
        edge_set = set(edges)
        
        # Map edge to index
        edge_index = {e: i for i, e in enumerate(edges)}
        
        for i, (u, v) in enumerate(edges):
            # Check all other edges
            # Optimization: Only check edges incident to u or v, or adjacent to u or v
            # Neighbors of u and v in G
            candidates = set()
            candidates.update(adj[u])
            candidates.update(adj[v])
            
            # Edges sharing u or v
            for w in adj[u]:
                e1 = (min(u, w), max(u, w))
                if e1 in edge_index:
                    conflict[i].add(edge_index[e1])
            for w in adj[v]:
                e1 = (min(v, w), max(v, w))
                if e1 in edge_index:
                    conflict[i].add(edge_index[e1])
            
            # Edges connected by an edge (u-x, v-y etc)
            # An edge (x,y) is conflicting if x in {u,v} or y in {u,v} (handled above)
            # OR if (u,x) in E, (u,y) in E, (v,x) in E, (v,y) in E
            # So for each neighbor x of u, and neighbor y of v, check if (x,y) is an edge?
            # No, the condition is: edge (u,v) and edge (x,y) are connected by an edge.
            # This means there is an edge in G connecting a vertex in {u,v} to a vertex in {x,y}.
            # Since (u,v) and (x,y) are distinct and don't share vertices (if they did, they'd be in the first check),
            # we need to check if any of (u,x), (u,y), (v,x), (v,y) is in E.
            
            # Let's iterate through all edges for simplicity first, m is small (~120)
            # m^2 is 14400, which is very fast.
            for j in range(m):
                if i == j: continue
                x, y = edges[j]
                # Check shared vertex
                if u == x or u == y or v == x or v == y:
                    conflict[i].add(j)
                    continue
                # Check connected by edge
                if (u, x) in edge_set or (u, y) in edge_set or (v, x) in edge_set or (v, y) in edge_set:
                    conflict[i].add(j)
        
        # Symmetrize
        for i in range(m):
            for j in conflict[i]:
                if i not in conflict[j]:
                    conflict[j].add(i)

        # Greedy Initialization (DSATUR-like but simpler: Random Order + Greedy)
        # We will use a local search to improve.
        
        def get_initial_coloring():
            colors = [-1] * m
            # Order by degree descending
            order = sorted(range(m), key=lambda i: -len(conflict[i]))
            for i in order:
                used = set()
                for j in conflict[i]:
                    if colors[j] != -1:
                        used.add(colors[j])
                c = 0
                while c in used:
                    c += 1
                colors[i] = c
            return colors

        best_colors = get_initial_coloring()
        best_max = max(best_colors) + 1
        
        # Local Search: Try to reduce max color
        # We maintain a valid coloring.
        # Move: Pick a random edge i, try to change its color to a smaller color c that is valid.
        # If we can't find a smaller color, try to swap colors of i and j to allow a third edge k to use a smaller color?
        # Simpler: Just try to recolor to any valid color, and if it reduces the global max, keep it.
        # To escape local minima, we might need to increase colors temporarily? 
        # For small graphs, simple hill climbing with random restarts is often enough.
        
        import time
        start_time = time.time()
        timeout = 10.0 # per instance, total 30s
        
        for restart in range(5):
            colors = get_initial_coloring()
            current_max = max(colors) + 1
            
            while time.time() - start_time < timeout:
                improved = False
                # Try to reduce max color
                if current_max > 1:
                    # Try to remove the highest color
                    highest_color = current_max - 1
                    edges_with_highest = [i for i in range(m) if colors[i] == highest_color]
                    
                    if edges_with_highest:
                        # Try to recolor one of these edges to a lower color
                        random.shuffle(edges_with_highest)
                        for i in edges_with_highest:
                            # Try colors 0 to highest_color - 1
                            for c in range(highest_color):
                                # Check if c is valid for i
                                valid = True
                                for j in conflict[i]:
                                    if colors[j] == c:
                                        valid = False
                                        break
                                if valid:
                                    colors[i] = c
                                    current_max -= 1
                                    improved = True
                                    break
                            if improved:
                                break
                
                if not improved:
                    # Stuck at this max. Try to shuffle to find a new path?
                    # Or just break and restart
                    break
            
            if current_max < best_max:
                best_max = current_max
                best_colors = colors[:]
            
            if best_max == 1: # Can't do better
                break
                
        # Compact colors
        # The colors in best_colors might be sparse (e.g., 0, 1, 3). We need to return a list of colors.
        # The verifier checks "total number of colours used".
        # Usually this means the size of the set of colors used.
        # We should relabel them to 0..k-1 to be safe, although the verifier might just count distinct values.
        # Let's relabel to be compact.
        
        color_map = {}
        next_color = 0
        final_colors = [0] * m
        for i in range(m):
            c = best_colors[i]
            if c not in color_map:
                color_map[c] = next_color
                next_color += 1
            final_colors[i] = color_map[c]
            
        return final_colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        colors = solve_instance(edges, adj)
        out.append(colors)
    return out
# EVOLVE-BLOCK-END
