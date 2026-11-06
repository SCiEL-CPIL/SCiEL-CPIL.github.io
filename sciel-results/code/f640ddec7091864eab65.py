"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

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

    def solve_instance(edges, max_colors_hint):
        """
        Find minimum number of colours such that each colour class is an induced matching.
        Uses branch-and-bound on the conflict graph (edges that cannot share a colour).
        """
        m = len(edges)
        if m == 0:
            return []

        # Build adjacency: two edges conflict if they share a vertex or are joined by an edge
        # Edge i = (u_i, v_i). Conflict if:
        #   - share a vertex: {u_i, v_i} & {u_j, v_j} != empty
        #   - joined by an edge: (u_i, u_j) in edge_set or (u_i, v_j) in edge_set or (v_i, u_j) in edge_set or (v_i, v_j) in edge_set
        edge_set = set(edges)
        # For fast lookup of whether (a,b) is an edge
        def is_edge(a, b):
            if a > b:
                a, b = b, a
            return (a, b) in edge_set

        # Build conflict bitmask for each edge
        # conflict[i] is a bitmask of edges that conflict with edge i
        conflict = [0] * m
        for i in range(m):
            u_i, v_i = edges[i]
            for j in range(i + 1, m):
                u_j, v_j = edges[j]
                # Check if they share a vertex
                if u_i == u_j or u_i == v_j or v_i == u_j or v_i == v_j:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                else:
                    # Check if joined by an edge
                    if is_edge(u_i, u_j) or is_edge(u_i, v_j) or is_edge(v_i, u_j) or is_edge(v_i, v_j):
                        conflict[i] |= (1 << j)
                        conflict[j] |= (1 << i)

        # Greedy lower bound: process edges in order of descending conflict degree
        # and assign colours greedily
        def greedy_coloring(order):
            colour = [-1] * m
            used = 0
            for i in order:
                # Find first available colour
                for c in range(used + 1):
                    # Check if colour c is available for edge i
                    # i.e., no edge with colour c conflicts with i
                    ok = True
                    # We need to check all edges with colour c
                    # This is expensive; instead track per-colour conflict masks
                    pass
                # Simpler: just try colours in order
                assigned = False
                for c in range(used):
                    # Check if any edge with colour c conflicts with i
                    conflict_with_c = 0
                    # We don't have per-colour tracking here, so do it naively
                    # Actually let's use a different approach
                    pass
                # Let's just do simple greedy: for each edge in order, try colour 0,1,2,...
                for c in range(used + 1):
                    valid = True
                    # Check all previously assigned edges with colour c
                    for j in range(m):
                        if colour[j] == c:
                            if conflict[i] & (1 << j):
                                valid = False
                                break
                    if valid:
                        colour[i] = c
                        assigned = True
                        break
                if not assigned:
                    colour[i] = used
                    used += 1
            return colour, used

        # Order edges by descending conflict degree for better greedy
        order = sorted(range(m), key=lambda i: bin(conflict[i]).count('1'), reverse=True)
        greedy_colour, greedy_count = greedy_coloring(order)

        # Now try to improve with branch and bound
        # We want to find a colouring with fewer colours than greedy_count
        best_colour = greedy_colour[:]
        best_count = greedy_count

        # Try to find a valid colouring with k colours for k = greedy_count-1, greedy_count-2, ...
        # For each k, check if the conflict graph can be coloured with k colours
        # where adjacent vertices (conflicting edges) must have different colours
        # This is standard graph colouring on the conflict graph

        # Use backtracking with bitsets
        # vertices = edges, edges in conflict graph = conflicts
        # We want to assign colours 0..k-1 to vertices such that no two adjacent have same colour

        def can_color_with_k(k, deadline):
            """Try to colour the conflict graph with k colours using backtracking."""
            if k >= m:
                return list(range(m))

            # Order vertices by descending degree (most constrained first)
            order_bb = sorted(range(m), key=lambda i: bin(conflict[i]).count('1'), reverse=True)

            colour = [-1] * m
            # colour_mask[c] = bitmask of vertices already assigned colour c
            colour_mask = [0] * k

            def backtrack(idx):
                if time.time() > deadline:
                    return None
                if idx == m:
                    return colour[:]

                v = order_bb[idx]
                # Try each colour
                for c in range(k):
                    if colour_mask[c] & (1 << v):
                        continue
                    # Check if v conflicts with any vertex already coloured c
                    if conflict[v] & colour_mask[c]:
                        continue
                    colour[v] = c
                    colour_mask[c] |= (1 << v)
                    result = backtrack(idx + 1)
                    if result is not None:
                        return result
                    colour[v] = -1
                    colour_mask[c] &= ~(1 << v)
                return None

            return backtrack(0)

        # Try to reduce the number of colours
        for k in range(greedy_count - 1, 0, -1):
            if time.time() > DEADLINE - 10:
                break
            result = can_color_with_k(k, DEADLINE - 5)
            if result is not None:
                best_colour = result
                best_count = k
                break

        # Convert colour array (indexed by edge index in original sorted order) to list
        # Wait, order_bb reorders, but colour is indexed by original vertex index
        # So best_colour[i] is the colour of edges[i]
        return best_colour

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > DEADLINE - 5:
            # Not enough time, return best effort
            out.append([])
            continue
        edges = instance(*spec)
        d = spec[1]
        hint = 5 * d * d // 4 + 1  # slight above conjecture bound
        colours = solve_instance(edges, hint)
        out.append(colours)

    return out
# EVOLVE-BLOCK-END
