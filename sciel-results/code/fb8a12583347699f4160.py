"""Strong edge colourings of three fixed bounded-degree graphs."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

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

    def solve_one(n, d, seed):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Build adjacency for vertices
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Build conflict graph: two edges conflict if they share a vertex or are joined by an edge
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        conflict = [set() for _ in range(m)]
        
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i].add(j)
                    conflict[j].add(i)
                else:
                    if u2 in adj.get(u1, set()) or v2 in adj.get(u1, set()) or \
                       u2 in adj.get(v1, set()) or v2 in adj.get(v1, set()):
                        conflict[i].add(j)
                        conflict[j].add(i)
        
        # Convert to list of lists for faster access
        conflict_list = [list(s) for s in conflict]
        
        # Greedy coloring for upper bound
        def greedy_coloring():
            order = sorted(range(m), key=lambda i: -len(conflict_list[i]))
            color_of = [-1] * m
            used = 0
            for idx in order:
                forbidden = set()
                for c_idx in conflict_list[idx]:
                    if color_of[c_idx] != -1:
                        forbidden.add(color_of[c_idx])
                c = 0
                while c in forbidden:
                    c += 1
                color_of[idx] = c
                if c >= used:
                    used = c + 1
            return used, color_of

        ub, _ = greedy_coloring()
        
        # B&B to find exact minimum
        # We try k from 1 to ub
        # can_color(k)
        
        def can_color(k):
            if k == 0:
                return m == 0
            
            color_of = [-1] * m
            # saturation: count of distinct colors in neighbors
            saturation = [0] * m
            # For each vertex, track which colors are used by neighbors
            neighbor_colors = [set() for _ in range(m)]
            
            # DSATUR heuristic
            # We maintain a list of uncolored vertices
            uncolored = set(range(m))
            
            # Precompute degrees
            degrees = [len(c) for c in conflict_list]
            
            def get_best_vertex(uncolored_set):
                best_v = -1
                best_sat = -1
                best_deg = -1
                for v in uncolored_set:
                    sat = saturation[v]
                    deg = degrees[v]
                    if sat > best_sat or (sat == best_sat and deg > best_deg):
                        best_v = v
                        best_sat = sat
                        best_deg = deg
                return best_v

            def backtrack(uncolored_set):
                if not uncolored_set:
                    return True
                
                v = get_best_vertex(uncolored_set)
                if v == -1:
                    return False
                
                uncolored_set.remove(v)
                
                # Determine available colors
                available = [c for c in range(k) if c not in neighbor_colors[v]]
                
                # Sort available colors? Maybe try most constraining first?
                # For now, just iterate.
                
                for c in available:
                    color_of[v] = c
                    # Update neighbors
                    changed = []
                    for u in conflict_list[v]:
                        if u in uncolored_set:
                            if c not in neighbor_colors[u]:
                                neighbor_colors[u].add(c)
                                saturation[u] += 1
                                changed.append(u)
                    
                    if backtrack(uncolored_set):
                        return True
                    
                    # Undo
                    for u in changed:
                        neighbor_colors[u].remove(c)
                        saturation[u] -= 1
                    
                color_of[v] = -1
                uncolored_set.add(v)
                return False
            
            return backtrack(uncolored)

        # Try increasing k
        for k in range(1, ub + 1):
            if can_color(k):
                # Re-run to get the actual coloring for this k
                # We need to store the coloring. Let's modify can_color to return the coloring.
                break
        
        # Re-do to get coloring for the found k
        # Actually, let's just use the greedy one if B&B is too slow or complex to extract.
        # But B&B should be fast. Let's make can_color return the coloring.
        
        def find_coloring(k):
            if k == 0:
                return [] if m == 0 else None
            
            color_of = [-1] * m
            saturation = [0] * m
            neighbor_colors = [set() for _ in range(m)]
            uncolored = set(range(m))
            degrees = [len(c) for c in conflict_list]
            
            def get_best_vertex(uncolored_set):
                best_v = -1
                best_sat = -1
                best_deg = -1
                for v in uncolored_set:
                    sat = saturation[v]
                    deg = degrees[v]
                    if sat > best_sat or (sat == best_sat and deg > best_deg):
                        best_v = v
                        best_sat = sat
                        best_deg = deg
                return best_v

            def backtrack(uncolored_set):
                if not uncolored_set:
                    return True
                
                v = get_best_vertex(uncolored_set)
                if v == -1:
                    return False
                
                uncolored_set.remove(v)
                
                available = [c for c in range(k) if c not in neighbor_colors[v]]
                
                for c in available:
                    color_of[v] = c
                    changed = []
                    for u in conflict_list[v]:
                        if u in uncolored_set:
                            if c not in neighbor_colors[u]:
                                neighbor_colors[u].add(c)
                                saturation[u] += 1
                                changed.append(u)
                    
                    if backtrack(uncolored_set):
                        return True
                    
                    for u in changed:
                        neighbor_colors[u].remove(c)
                        saturation[u] -= 1
                    color_of[v] = -1
                uncolored_set.add(v)
                return False
            
            if backtrack(uncolored):
                return color_of
            return None

        # Find min k
        min_k = ub
        for k in range(1, ub + 1):
            if can_color(k):
                min_k = k
                break
        
        coloring = find_coloring(min_k)
        if coloring is None:
            # Fallback to greedy
            _, coloring = greedy_coloring()
            
        return coloring

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        out.append(solve_one(*spec))
    return out
# EVOLVE-BLOCK-END
