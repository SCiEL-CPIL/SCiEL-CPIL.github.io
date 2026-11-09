"""Strong edge colourings of three fixed bounded-degree graphs."""
# EVOLVE-BLOCK-START
import sys
import time
sys.setrecursionlimit(10000)

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

    def solve_one(n, d, seed):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Build adjacency for vertices
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        # Build conflict graph
        conflict = [set() for _ in range(m)]
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                # Check if they share a vertex
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    continue
                # Check if there's an edge between a vertex of e1 and a vertex of e2
                if u2 in adj.get(u1, set()) or v2 in adj.get(u1, set()) or \
                   u2 in adj.get(v1, set()) or v2 in adj.get(v1, set()):
                    conflict[i].add(j)
                    conflict[j].add(i)
        
        # Convert conflict sets to lists for faster iteration
        conflict_lists = [list(s) for s in conflict]
        degrees = [len(lst) for lst in conflict_lists]

        # 1. Get initial upper bound using DSatur heuristic
        def dsatur_coloring(conflict_lists, degrees):
            n_vertices = len(conflict_lists)
            colors = [-1] * n_vertices
            uncolored = set(range(n_vertices))
            max_colors = 0
            # neighbor_colors: set of colors used by neighbors
            neighbor_colors = [set() for _ in range(n_vertices)]
            
            while uncolored:
                best_v = -1
                best_sat = -1
                best_deg = -1
                for v in uncolored:
                    sat = len(neighbor_colors[v])
                    if sat > best_sat or (sat == best_sat and degrees[v] > best_deg):
                        best_sat = sat
                        best_deg = degrees[v]
                        best_v = v
                
                if best_v == -1:
                    break
                    
                forbidden = neighbor_colors[best_v]
                c = 0
                while c in forbidden:
                    c += 1
                
                colors[best_v] = c
                if c >= max_colors:
                    max_colors = c + 1
                uncolored.remove(best_v)
                
                # Update neighbor colors
                for u in conflict_lists[best_v]:
                    neighbor_colors[u].add(c)
                
            return colors, max_colors

        init_colors, init_bound = dsatur_coloring(conflict_lists, degrees)
        
        # 2. Branch and Bound with Dynamic Ordering (DSATUR) and Forward Checking
        start_time = time.time()
        time_limit = 30.0
        
        best_colors = init_colors[:]
        best_bound = init_bound
        
        # State:
        # colors: list of -1 or color index
        # neighbor_colors: list of sets
        # uncolored: set of uncolored vertices
        
        def b_and_b(uncolored, colors, neighbor_colors, max_color_used):
            nonlocal best_bound, best_colors, start_time, time_limit
            
            if time.time() - start_time > time_limit:
                return False
            
            if not uncolored:
                if max_color_used < best_bound:
                    best_bound = max_color_used
                    best_colors = colors[:]
                return True
            
            # Pruning: if we already exceed best_bound, stop
            # Note: max_color_used is the number of colors used so far (0-indexed max is max_color_used-1, so count is max_color_used)
            # If we are about to use color max_color_used, total will be max_color_used + 1?
            # In DSATUR, c is 0-indexed. If we use c, and c == max_color_used, then we are adding a new color.
            # So total colors will be at least max_color_used + 1 if we use a new color, or max_color_used if we reuse.
            # Actually, max_color_used tracks the *next* available index? No, let's track `num_colors_used`.
            # Let's stick to `max_color_index` used.
            
            # Lower Bound: 
            # If any uncolored vertex has saturation == max_color_index + 1, it needs a new color.
            # But we don't know if we *will* use a new color yet.
            # Simple bound: if max_color_index + 1 > best_bound, we can't improve.
            # Actually, if we are at a node where we have used colors 0..k-1, and we are about to assign a vertex.
            # If k >= best_bound, prune.
            
            if max_color_used >= best_bound:
                return False

            # Select vertex: Max Saturation, then Max Degree
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in uncolored:
                sat = len(neighbor_colors[v])
                if sat > best_sat or (sat == best_sat and degrees[v] > best_deg):
                    best_sat = sat
                    best_deg = degrees[v]
                    best_v = v
            
            if best_v == -1:
                return False

            forbidden = neighbor_colors[best_v]
            
            # Try existing colors
            for c in range(max_color_used):
                if c not in forbidden:
                    colors[best_v] = c
                    uncolored.remove(best_v)
                    
                    # Forward Checking
                    valid = True
                    updates = []
                    for u in conflict_lists[best_v]:
                        if u in uncolored:
                            neighbor_colors[u].add(c)
                            updates.append(u)
                            # Check if u is now stuck
                            if len(neighbor_colors[u]) == max_color_used:
                                # All colors 0..max_color_used-1 are forbidden for u
                                # u will need color max_color_used
                                # This is not an immediate conflict unless max_color_used >= best_bound
                                # But it restricts future choices.
                                pass
                    
                    if valid:
                        if b_and_b(uncolored, colors, neighbor_colors, max_color_used):
                            # Undo
                            for u in updates:
                                neighbor_colors[u].remove(c)
                                uncolored.add(u)
                            colors[best_v] = -1
                            return True
                    
                    # Undo
                    for u in updates:
                        neighbor_colors[u].remove(c)
                    uncolored.add(u)
                    colors[best_v] = -1
            
            # Try new color
            if max_color_used < best_bound:
                c = max_color_used
                colors[best_v] = c
                uncolored.remove(best_v)
                
                valid = True
                updates = []
                for u in conflict_lists[best_v]:
                    if u in uncolored:
                        neighbor_colors[u].add(c)
                        updates.append(u)
                
                if valid:
                    if b_and_b(uncolored, colors, neighbor_colors, max_color_used + 1):
                        # Undo
                        for u in updates:
                            neighbor_colors[u].remove(c)
                            uncolored.add(u)
                        colors[best_v] = -1
                        return True
                
                # Undo
                for u in updates:
                    neighbor_colors[u].remove(c)
                uncolored.add(u)
                colors[best_v] = -1
            
            return False

        uncolored = set(range(m))
        colors = [-1] * m
        neighbor_colors = [set() for _ in range(m)]
        
        b_and_b(uncolored, colors, neighbor_colors, 0)
        
        return best_colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        out.append(solve_one(*spec))
    return out
# EVOLVE-BLOCK-END
