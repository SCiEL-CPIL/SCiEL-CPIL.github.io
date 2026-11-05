"""Strong edge colouring of bounded-degree graphs using DSATUR with bitset conflicts."""
import time
from random import Random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

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

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Map edge index
        edge_idx = {}
        for i, e in enumerate(edges):
            edge_idx[e] = i

        # Build conflict graph using bitmasks
        # Two edges conflict if they share a vertex or are joined by an edge
        # For each edge i = (u, v), conflicts with edge j = (a, b) if:
        #   - {u,v} ∩ {a,b} ≠ ∅ (share a vertex), OR
        #   - (u,a) ∈ E, (u,b) ∈ E, (v,a) ∈ E, or (v,b) ∈ E (joined by an edge)

        # Precompute for each vertex, the set of edges incident to it
        # And for each vertex, the set of edges incident to its neighbors
        edge_to_vertices = [(u, v) for u, v in edges]

        # For each edge, compute its conflict set as a bitmask
        # m can be up to ~240, so we can use Python integers as bitmasks
        conflict = [0] * m

        # For each edge i, find all edges j that conflict with it
        # An edge j=(a,b) conflicts with edge i=(u,v) if:
        #   j shares a vertex with i: a==u or a==v or b==u or b==v
        #   OR j is joined to i: (u,a) in adj[u], (u,b) in adj[u], (v,a) in adj[v], (v,b) in adj[v]

        # Build a mapping from vertex to list of incident edge indices
        vert_edges = [[] for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            vert_edges[u].append(i)
            vert_edges[v].append(i)

        # For each edge, compute its conflict bitmask
        for i, (u, v) in enumerate(edges):
            # Edges sharing vertex u or v
            for j in vert_edges[u]:
                if j != i:
                    conflict[i] |= (1 << j)
            for j in vert_edges[v]:
                if j != i:
                    conflict[i] |= (1 << j)

            # Edges joined by an edge: for each neighbor of u, edges incident to that neighbor
            for nu in adj[u]:
                for j in vert_edges[nu]:
                    if j != i:
                        conflict[i] |= (1 << j)
            for nv in adj[v]:
                for j in vert_edges[nv]:
                    if j != i:
                        conflict[i] |= (1 << j)

        # DSATUR greedy coloring
        # color[i] = -1 means uncolored
        color = [-1] * m
        # For each color c, maintain a bitmask of edges with that color
        color_masks = {}  # c -> bitmask
        # For each edge, maintain the set of colors used by its neighbors
        # We'll recompute this each step for simplicity, using bitmasks

        # Track available colors: start with color 0
        next_color = 0

        for step in range(m):
            # Choose uncolored edge with maximum saturation (number of distinct colors in its neighborhood)
            # Tie-break: highest degree in conflict graph
            best_i = -1
            best_sat = -1
            best_deg = -1

            for i in range(m):
                if color[i] != -1:
                    continue
                # Compute saturation: distinct colors among neighbors
                # Neighbors are edges j where conflict[i] has bit j set
                # We need to check which colors are used by those neighbors
                # Efficient: for each color c, check if color_masks[c] & conflict[i] != 0
                sat = 0
                for c in color_masks:
                    if color_masks[c] & conflict[i]:
                        sat += 1
                deg = bin(conflict[i]).count('1')

                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_i = i

            if best_i == -1:
                break

            # Assign the smallest available color to best_i
            # A color c is available if no neighbor of best_i has color c
            # i.e., color_masks[c] & conflict[best_i] == 0
            assigned_color = None
            for c in range(next_color + 1):
                if c not in color_masks or (color_masks[c] & conflict[best_i]) == 0:
                    assigned_color = c
                    break

            if assigned_color is None:
                assigned_color = next_color
                next_color += 1

            color[best_i] = assigned_color
            if assigned_color not in color_masks:
                color_masks[assigned_color] = 1 << best_i
            else:
                color_masks[assigned_color] |= (1 << best_i)

        return color

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline:
            # Fallback: return a trivial valid coloring (each edge gets its own color)
            edges = instance(*spec)
            out.append(list(range(len(edges))))
            continue
        out.append(solve_instance(*spec))

    return out

# EVOLVE-BLOCK-END
