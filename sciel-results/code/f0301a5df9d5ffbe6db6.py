"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def construct():
    n = P['n']
    edges = P['edges']
    start_time = time.time()
    deadline = 85.0  # safety margin for 90s wall-clock limit

    # Build adjacency list
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Order vertices by degree (descending) for better backtracking
    vertex_order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    # Map from vertex to position in order
    vertex_pos = {v: i for i, v in enumerate(vertex_order)}

    def solve_with_k(k):
        """Try to find a valid coloring with exactly k colors using backtracking."""
        if time.time() > deadline - 2:
            return None

        # color[v] = assigned color or -1 if unassigned
        color = [-1] * n

        # For each pair of colors (a, b) with a < b, maintain a DSU
        # dsu[a][b] = parent array for the subgraph induced by colors a and b
        # We only need to track edges where both endpoints have colors in {a, b}
        # and at least one edge connects them.

        # Actually, let's think more carefully. The subgraph induced by colors {a, b}
        # contains all vertices colored a or b, and all edges between them.
        # We need this subgraph to be a forest.
        # Using DSU: when we assign a color to a vertex, we add edges from this vertex
        # to already-colored neighbors that have the same or the other color in the pair.

        # Let's use a simpler approach: maintain for each color pair (a,b) a DSU.
        # When we color vertex v with color c, for each already-colored neighbor u:
        #   Let cu = color[u]
        #   The relevant color pairs to check:
        #     If cu == c: no new edge in any pair (same color, not bicolored)
        #     If cu != c: the pair (min(c, cu), max(c, cu)) gets a new edge (u, v)
        #   For that pair, check if u and v are already connected in the DSU.
        #   If yes, this would create a cycle -> invalid.
        #   If no, union them in the DSU.

        # DSU structure: for each pair (a, b), a, b in range(k), a < b
        # We'll use a dictionary mapping (a, b) to a parent list
        # But that's expensive. Let's use a flat structure.

        # Alternative: for each vertex v and each color c != color[v] that v is adjacent to,
        # we can precompute constraints. But let's just do the DSU approach.

        # Initialize DSUs for all color pairs
        # dsu is a dict: key = (a, b) with a < b, value = list of parents
        dsu = {}

        def find(parents, x):
            while parents[x] != x:
                parents[x] = parents[parents[x]]
                x = parents[x]
            return x

        def union(parents, x, y):
            rx, ry = find(parents, x), find(parents, y)
            if rx == ry:
                return False  # already connected, would create cycle
            parents[ry] = rx
            return True  # successfully merged, no cycle

        # Initialize DSU for each pair
        for a in range(k):
            for b in range(a + 1, k):
                dsu[(a, b)] = list(range(n))

        # Backtracking search
        def backtrack(idx):
            if time.time() > deadline - 1:
                return None

            if idx == n:
                return True

            v = vertex_order[idx]

            # Try each color
            for c in range(k):
                # Check if color c is valid for vertex v
                valid = True

                # Check adjacent vertices
                for u in adj[v]:
                    if color[u] != -1:
                        if color[u] == c:
                            valid = False
                            break
                        # Check bicolored cycle constraint
                        a, b = min(c, color[u]), max(c, color[u])
                        key = (a, b)
                        if key in dsu:
                            parents = dsu[key]
                            if find(parents, u) == find(parents, v):
                                valid = False
                                break
                            # Temporarily union
                            parents[find(parents, v)] = find(parents, u)
                            # We need to undo this later if we backtrack
                            # Let's save the state
                            # Actually, let's use a different approach:
                            # Instead of modifying in place, let's collect all changes
                            # and undo them.

                if not valid:
                    continue

                # If we got here, color c seems valid. But we modified DSUs above.
                # Let's redo this properly with save/restore.

                # Actually, let me restructure: for each candidate color, 
                # first check validity without modifying, then modify and recurse.

                # Re-check: I made an error above by modifying DSU during checking.
                # Let me restart the approach.

                pass

            return False

        # Let me rewrite the backtracking more carefully

        def backtrack2(idx):
            if time.time() > deadline - 1:
                return None

            if idx == n:
                return True

            v = vertex_order[idx]

            # Get valid colors for v
            valid_colors = []
            for c in range(k):
                ok = True
                for u in adj[v]:
                    if color[u] != -1:
                        if color[u] == c:
                            ok = False
                            break
                        a, b = min(c, color[u]), max(c, color[u])
                        if (a, b) in dsu:
                            ru = find(dsu[(a, b)], u)
                            rv = find(dsu[(a, b)], v)
                            if ru == rv:
                                ok = False
                                break
                if ok:
                    valid_colors.append(c)

            # Try each valid color
            for c in valid_colors:
                color[v] = c

                # Record DSU changes for rollback
                changes = []
                for u in adj[v]:
                    if color[u] != -1:
                        a, b = min(c, color[u]), max(c, color[u])
                        key = (a, b)
                        if key in dsu:
                            parents = dsu[key]
                            ru = find(parents, u)
                            rv = find(parents, v)
                            if ru != rv:
                                changes.append((key, rv, ru))  # (key, child_root, parent_root)
                                parents[rv] = ru

                if backtrack2(idx + 1):
                    return True

                # Rollback DSU changes
                for key, rv, ru in changes:
                    dsu[key][rv] = rv  # restore parent of rv to itself

                color[v] = -1

            return False

        result = backtrack2(0)
        if result:
            return color
        return None

    # Try increasing number of colors
    for k in range(3, 10):
        if time.time() > deadline - 3:
            break
        result = solve_with_k(k)
        if result is not None:
            # Verify the solution
            valid = True
            for u, v in edges:
                if result[u] == result[v]:
                    valid = False
                    break
            if valid:
                # Check bicolored cycles
                for a in range(k):
                    for b in range(a + 1, k):
                        parents = list(range(n))
                        for u, v in edges:
                            cu, cv = result[u], result[v]
                            if (cu == a and cv == b) or (cu == b and cv == a):
                                ru, rv = find(parents, u), find(parents, v)
                                if ru == rv:
                                    valid = False
                                    break
                                parents[rv] = ru
                            elif (cu == b and cv == a):
                                ru, rv = find(parents, u), find(parents, v)
                                if ru == rv:
                                    valid = False
                                    break
                                parents[rv] = ru
                        if not valid:
                            break
                    if not valid:
                        break
            if valid:
                return [int(x) for x in result]

    # Fallback: return trivial solution
    return list(range(n))

# EVOLVE-BLOCK-END
