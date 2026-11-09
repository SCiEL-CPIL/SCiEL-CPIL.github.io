"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin under 150s kill

    n = 20
    num_colors = 20

    def is_valid_addition(u, v, c, adj_set, col_dict):
        """Check if adding edge (u,v) with color c is valid.
        adj_set[x] is a dict: adj_set[x][y] = color of edge (x,y)
        col_dict[(min,max)] = color
        """
        for a in range(n):
            if a == u or a == v:
                continue
            ua_exists = a in adj_set[u]
            for b in range(n):
                if b == u or b == v or b == a:
                    continue
                # Type A: 4-cycle u-a-b-v-u, edges (u,a),(a,b),(b,v),(v,u)
                if ua_exists and b in adj_set[a] and v in adj_set[b]:
                    c1 = col_dict[(min(u,a), max(u,a))]
                    c2 = col_dict[(min(a,b), max(a,b))]
                    c3 = col_dict[(min(b,v), max(b,v))]
                    if len({c1, c2, c3, c}) != 4:
                        return False
                # Type B: 4-cycle u-b-a-v-u, edges (u,b),(b,a),(a,v),(v,u)
                if b in adj_set[u] and a in adj_set[b] and v in adj_set[a]:
                    c1 = col_dict[(min(u,b), max(u,b))]
                    c2 = col_dict[(min(a,b), max(a,b))]
                    c3 = col_dict[(min(a,v), max(a,v))]
                    if len({c1, c2, c3, c}) != 4:
                        return False
        return True

    # Generate all possible edges
    all_edges = list(itertools.combinations(range(n), 2))

    # Multiple restarts with random edge ordering
    rng = random.Random(42)

    best_edges = None
    best_count = 0

    for restart in range(100):
        if time.time() - t0 > DEADLINE - 5:  # leave 5s for verification
            break

        # Shuffle edge order
        edges_order = all_edges[:]
        rng.shuffle(edges_order)

        # Initialize empty graph
        adj_set = [dict() for _ in range(n)]
        col_dict = {}
        current_edges = []

        for (u, v) in edges_order:
            if time.time() - t0 > DEADLINE - 5:
                break

            # Try each color (randomize order for diversity)
            valid_color = None
            color_order = list(range(num_colors))
            rng.shuffle(color_order)

            for c in color_order:
                if is_valid_addition(u, v, c, adj_set, col_dict):
                    valid_color = c
                    break

            if valid_color is not None:
                # Add edge
                adj_set[u][v] = valid_color
                adj_set[v][u] = valid_color
                key = (min(u, v), max(u, v))
                col_dict[key] = valid_color
                current_edges.append([u, v, valid_color])

        # Update best
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]

    # If we found something better than the 1-factorization (190 edges), verify and use it
    if best_count >= 190 and best_edges is not None:
        # Verify the best solution
        col = {}
        for u, v, c in best_edges:
            key = (min(u, v), max(u, v))
            col[key] = c

        valid = True
        for quad in itertools.combinations(range(n), 4):
            a, b, c, d = quad
            for (v1, v2, v3, v4) in [(a, b, c, d), (a, b, d, c), (a, c, b, d)]:
                e1 = col.get((min(v1, v2), max(v1, v2)))
                e2 = col.get((min(v2, v3), max(v2, v3)))
                e3 = col.get((min(v3, v4), max(v3, v4)))
                e4 = col.get((min(v4, v1), max(v4, v1)))
                if e1 is None or e2 is None or e3 is None or e4 is None:
                    continue
                if len({e1, e2, e3, e4}) != 4:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            return best_edges

    # Fallback: 1-factorization of K20 (190 edges)
    edges = []
    for r in range(19):
        colour = r
        edges.append([19, r, colour])
        for i in range(1, 10):
            u = (r + i) % 19
            v = (r - i) % 19
            edges.append([u, v, colour])

    return edges
# EVOLVE-BLOCK-END
