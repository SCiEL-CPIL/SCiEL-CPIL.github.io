"""Strong edge colouring via iterative induced matching extraction."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def strong_coloring(edges, adj, n, max_time=135):
        """
        Strong edge coloring by iteratively extracting induced matchings.
        Each color class is an induced matching.
        """
        if not edges:
            return []
        
        edge_set = set(edges)
        color = {}
        color_idx = 0
        remaining = set(edges)
        
        # Precompute for each edge, the set of vertices in its "closed neighborhood"
        # Two edges conflict if their closed neighborhoods intersect
        # Edge (u,v) has closed neighborhood {u,v} ∪ N(u) ∪ N(v)
        edge_neighborhood = {}
        for e in edges:
            u, v = e
            near = {u, v}
            for x in (u, v):
                near.update(adj.get(x, set()))
            edge_neighborhood[e] = near
        
        while remaining:
            if time.time() - start_time > max_time:
                # Fallback: greedy color the rest
                for e in sorted(remaining):
                    u, v = e
                    near = edge_neighborhood[e]
                    taken = set()
                    for f, c in color.items():
                        if edge_neighborhood[f] & near:
                            taken.add(c)
                    c = 0
                    while c in taken:
                        c += 1
                    color[e] = c
                remaining = set()
                break
            
            # Greedy: find a large induced matching
            # Sort remaining edges by degree (fewer conflicts first)
            remaining_list = sorted(remaining, key=lambda e: len(edge_neighborhood[e]))
            
            matching = []
            used_vertices = set()
            
            for e in remaining_list:
                if e not in remaining:
                    continue
                u, v = e
                # Check if e can be added to the matching
                # It must not share vertices with any edge in matching
                # And no edge in the graph should connect an endpoint of e to an endpoint of a matching edge
                can_add = True
                if u in used_vertices or v in used_vertices:
                    can_add = False
                else:
                    # Check if any edge connects u or v to any vertex in used_vertices
                    for x in (u, v):
                        for y in used_vertices:
                            if (min(x, y), max(x, y)) in edge_set:
                                can_add = False
                                break
                        if not can_add:
                            break
                
                if can_add:
                    matching.append(e)
                    used_vertices.add(u)
                    used_vertices.add(v)
            
            if not matching:
                matching = [min(remaining)]
            
            # Assign this color class
            for e in matching:
                color[e] = color_idx
            remaining -= set(matching)
            color_idx += 1
        
        return [color[e] for e in edges]

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() - start_time > DEADLINE:
            edges = instance(*spec)
            out.append(list(range(len(edges))))
            continue
        
        n, d, seed = spec
        edges = instance(n, d, seed)
        
        # Build adjacency
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        colors = strong_coloring(edges, adj, n)
        out.append(colors)
    
    return out
# EVOLVE-BLOCK-END
