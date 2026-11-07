"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard: evaluator kills after 150s; use 140s as safety margin
    deadline = time.time() + 140.0

    def instance(splits, seed, palette=5):
        from random import Random
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_instance(n, edges, lists):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Precompute color sets as frozensets for fast membership
        color_sets = [frozenset(lst) for lst in lists]

        # Order vertices by descending degree (most constrained first)
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        # Map position in order -> vertex
        pos_in_order = [0] * n
        for i, v in enumerate(order):
            pos_in_order[v] = i

        # Assignments: -1 means unassigned
        assign = [-1] * n

        # For each vertex, track which colors are still available
        # We'll use a recursive backtracking with forward checking

        # To speed up, we'll use a stack-based approach with undo
        # But given the small size (n up to 408), recursion with backtracking should be fine

        # Use a more efficient approach: maintain domains as lists
        domains = [list(lst) for lst in lists]

        # We'll process vertices in order, but use backtracking
        # Key insight: bipartite graph, 3 colors, Alon-Tarsi guarantees solution

        # Let's use a simple backtracking with MRV (Minimum Remaining Values)
        # but constrained to the precomputed order for determinism and speed

        # Actually, let's use a hybrid: process in degree order, but if a vertex has
        # only 1 choice, commit immediately (unit propagation)

        # Simple backtracking with forward checking
        def backtrack(idx):
            # Check deadline
            if time.time() > deadline:
                return False

            if idx == n:
                return True

            v = order[idx]
            # Get available colors for v
            avail = []
            for c in color_sets[v]:
                # Check if c is not used by any already-assigned neighbor
                conflict = False
                for nb in adj[v]:
                    if assign[nb] == c:
                        conflict = True
                        break
                if not conflict:
                    avail.append(c)

            if not avail:
                return False

            # Try each color
            for c in avail:
                assign[v] = c
                if backtrack(idx + 1):
                    return True
                assign[v] = -1

            return False

        if backtrack(0):
            return assign

        # Fallback: try a different ordering or just return the greedy result
        # This shouldn't happen due to Alon-Tarsi, but just in case
        # Use greedy as fallback
        colour = [None] * n
        for v in order:
            taken = set()
            for nb in adj[v]:
                if colour[nb] is not None:
                    taken.add(colour[nb])
            free = [c for c in lists[v] if c not in taken]
            colour[v] = free[0] if free else lists[v][0]
        return colour

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        result = solve_instance(n, edges, lists)
        # Ensure we return Python ints in a list
        out.append([int(c) for c in result])

    return out
# EVOLVE-BLOCK-END
