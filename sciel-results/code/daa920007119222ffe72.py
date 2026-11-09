"""Strong edge colourings of three fixed bounded-degree graphs."""
# EVOLVE-BLOCK-START
import time
import random
import heapq

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit

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

    def build_adjacency(edges):
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        return adj

    def dsatur_strong_coloring(edges, adj):
        """DSATUR-like coloring for strong edge coloring."""
        n_edges = len(edges)
        if n_edges == 0:
            return []
        
        # For each edge, compute its "neighborhood" set of edges that conflict with it
        # Two edges conflict if they share a vertex or are joined by an edge
        edge_index = {e: i for i, e in enumerate(edges)}
        
        # Conflict: edges (u,v) and (x,y) conflict if:
        # 1. They share a vertex, OR
        # 2. There's an edge connecting an endpoint of one to an endpoint of the other
        # i.e., for edges e1=(a,b) and e2=(c,d), they conflict if:
        #    {a,b} & {c,d} != empty OR (a,c) in adj OR (a,d) in adj OR (b,c) in adj OR (b,d) in adj
        
        # Build conflict sets for each edge
        conflict = [set() for _ in range(n_edges)]
        for i, (u, v) in enumerate(edges):
            for j in range(i + 1, n_edges):
                x, y = edges[j]
                # Check if edges conflict
                if (u == x or u == y or v == x or v == y or
                    (u, x) in adj.get(u, set()) or (u, y) in adj.get(u, set()) or
                    (v, x) in adj.get(v, set()) or (v, y) in adj.get(v, set())):
                    conflict[i].add(j)
                    conflict[j].add(i)
        
        # DSATUR: pick edge with highest saturation (number of distinct colors in neighbors)
        # then highest degree in conflict graph
        color = [-1] * n_edges
        colors_used = 0
        
        # Priority queue: (-saturation, -degree, edge_index)
        # We'll use a list and re-sort each iteration for simplicity with small n
        
        uncolored = set(range(n_edges))
        
        while uncolored:
            if time.time() > deadline - 5:
                break  # Return what we have so far
            
            # Compute saturation and degree for each uncolored edge
            best_edge = -1
            best_sat = -1
            best_deg = -1
            
            for ei in uncolored:
                # Saturation: number of distinct colors used by colored neighbors
                neighbor_colors = set()
                for ni in conflict[ei]:
                    if color[ni] != -1:
                        neighbor_colors.add(color[ni])
                sat = len(neighbor_colors)
                deg = len(conflict[ei]) - sum(1 for ni in conflict[ei] if color[ni] == -1)
                
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_edge = ei
            
            if best_edge == -1:
                break
            
            # Find smallest available color
            neighbor_colors = set()
            for ni in conflict[best_edge]:
                if color[ni] != -1:
                    neighbor_colors.add(color[ni])
            
            c = 0
            while c in neighbor_colors:
                c += 1
            
            color[best_edge] = c
            if c >= colors_used:
                colors_used = c + 1
            uncolored.discard(best_edge)
        
        return color

    def local_search_improve(color, edges, adj, max_iters=200):
        """Try to reduce color count via moves and swaps."""
        n_edges = len(edges)
        if n_edges == 0:
            return color, 0
        
        # Build conflict sets
        conflict = [set() for _ in range(n_edges)]
        for i, (u, v) in enumerate(edges):
            for j in range(i + 1, n_edges):
                x, y = edges[j]
                if (u == x or u == y or v == x or v == y or
                    (u, x) in adj.get(u, set()) or (u, y) in adj.get(u, set()) or
                    (v, x) in adj.get(v, set()) or (v, y) in adj.get(v, set())):
                    conflict[i].add(j)
                    conflict[j].add(i)
        
        max_color = max(color) + 1 if color else 0
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            if time.time() > deadline - 5:
                break
            improved = False
            iters += 1
            
            # Try to move each edge to a lower color
            for ei in range(n_edges):
                if color[ei] == 0:
                    continue
                # Check if we can move to color 0..color[ei]-1
                for c in range(color[ei]):
                    can_move = True
                    for ni in conflict[ei]:
                        if color[ni] == c:
                            can_move = False
                            break
                    if can_move:
                        color[ei] = c
                        improved = True
                        break
            
            # Try to merge two color classes
            if max_color > 1:
                # Find pairs of colors that can be merged
                for c1 in range(max_color):
                    if time.time() > deadline - 5:
                        break
                    for c2 in range(c1 + 1, max_color):
                        # Check if all edges of color c1 can move to color c2 or vice versa
                        edges_c1 = [i for i in range(n_edges) if color[i] == c1]
                        edges_c2 = [i for i in range(n_edges) if color[i] == c2]
                        
                        if not edges_c1 or not edges_c2:
                            continue
                        
                        # Try moving all of c1 to c2
                        can_merge = True
                        for ei in edges_c1:
                            for ni in conflict[ei]:
                                if color[ni] == c2:
                                    can_merge = False
                                    break
                            if not can_merge:
                                break
                        
                        if can_merge:
                            for ei in edges_c1:
                                color[ei] = c2
                            max_color = c2  # c1 is now empty
                            improved = True
                            break
                    
                    if improved:
                        break
                
                if improved:
                    continue
            
            # Try swap: exchange colors of two edges
            for ei in range(n_edges):
                if time.time() > deadline - 5:
                    break
                for ej in range(ei + 1, n_edges):
                    if color[ei] == color[ej]:
                        continue
                    c1, c2 = color[ei], color[ej]
                    # Check if swapping is valid
                    valid = True
                    for ni in conflict[ei]:
                        if ni != ej and color[ni] == c2:
                            valid = False
                            break
                    if valid:
                        for ni in conflict[ej]:
                            if ni != ei and color[ni] == c1:
                                valid = False
                                break
                    if valid:
                        color[ei], color[ej] = c2, c1
                        improved = True
                        break
                if improved:
                    break
        
        final_max_color = max(color) + 1 if color else 0
        return color, final_max_color

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        adj = build_adjacency(edges)
        
        # Initial coloring via DSATUR
        color = dsatur_strong_coloring(edges, adj)
        
        # Local search improvement
        color, num_colors = local_search_improve(color, edges, adj)
        
        return [color[i] for i in range(len(edges))]

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline - 10:
            # Fallback: simple greedy
            edges = instance(*spec)
            adj = build_adjacency(edges)
            color = [0] * len(edges)
            for i, e in enumerate(edges):
                u, v = e
                near = {u, v}
                for x in (u, v):
                    near.update(adj.get(x, set()))
                taken = set()
                for j in range(i):
                    ej = edges[j]
                    if set(ej) & near:
                        taken.add(color[j])
                c = 0
                while c in taken:
                    c += 1
                color[i] = c
            out.append(color)
        else:
            out.append(solve_instance(*spec))
    
    return out
# EVOLVE-BLOCK-END
