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
        
        # 1. Get initial upper bound using DSatur heuristic
        def dsatur_coloring(conflict):
            n_vertices = len(conflict)
            colors = [-1] * n_vertices
            deg = [len(conflict[i]) for i in range(n_vertices)]
            uncolored = set(range(n_vertices))
            max_colors = 0
            
            while uncolored:
                best_v = -1
                best_sat = -1
                best_deg = -1
                for v in uncolored:
                    neigh_colors = set()
                    for u in conflict[v]:
                        if colors[u] != -1:
                            neigh_colors.add(colors[u])
                    sat = len(neigh_colors)
                    if sat > best_sat or (sat == best_sat and deg[v] > best_deg):
                        best_sat = sat
                        best_deg = deg[v]
                        best_v = v
                
                if best_v == -1:
                    break
                    
                forbidden = set()
                for u in conflict[best_v]:
                    if colors[u] != -1:
                        forbidden.add(colors[u])
                
                c = 0
                while c in forbidden:
                    c += 1
                
                colors[best_v] = c
                if c >= max_colors:
                    max_colors = c + 1
                uncolored.remove(best_v)
                
            return colors, max_colors

        init_colors, init_bound = dsatur_coloring(conflict)
        
        # 2. Branch and Bound to improve
        order = sorted(range(m), key=lambda i: -len(conflict[i]))
        
        start_time = time.time()
        time_limit = 30.0
        
        best_colors = init_colors[:]
        best_bound = init_bound
        
        def b_and_b(idx, current_colors, color_of):
            nonlocal best_bound, best_colors, start_time, time_limit
            
            if time.time() - start_time > time_limit:
                return False
            
            if idx == m:
                if current_colors < best_bound:
                    best_bound = current_colors
                    best_colors = color_of[:]
                return True
            
            if current_colors >= best_bound:
                return False
            
            v = order[idx]
            
            forbidden = set()
            for u in conflict[v]:
                if color_of[u] != -1:
                    forbidden.add(color_of[u])
            
            for c in range(current_colors):
                if c not in forbidden:
                    color_of[v] = c
                    if b_and_b(idx + 1, current_colors, color_of):
                        return True
                    color_of[v] = -1
            
            if current_colors < best_bound:
                color_of[v] = current_colors
                if b_and_b(idx + 1, current_colors + 1, color_of):
                    return True
                color_of[v] = -1
            
            return False
        
        color_of = [-1] * m
        b_and_b(0, 0, color_of)
        
        return best_colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        out.append(solve_one(*spec))
    return out
# EVOLVE-BLOCK-END
