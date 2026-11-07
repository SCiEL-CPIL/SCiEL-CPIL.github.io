"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(splits, seed, palette=5):
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

    def color_instance(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Use a randomized greedy approach with multiple restarts
        # For each restart, try a different random ordering and color choice
        rng = Random(12345)  # Fixed seed for determinism in the solver itself

        for attempt in range(200):
            # Random ordering of vertices
            order = list(range(n))
            rng.shuffle(order)

            assignment = [-1] * n
            valid = True

            for v in order:
                # Find colors in lists[v] that don't conflict with already-colored neighbors
                available = []
                for c in lists[v]:
                    conflict = False
                    for u in adj[v]:
                        if assignment[u] == c:
                            conflict = True
                            break
                    if not conflict:
                        available.append(c)

                if not available:
                    valid = False
                    break

                # Randomly pick a color from available
                assignment[v] = rng.choice(available)

            if valid:
                return [int(c) for c in assignment]

        # Fallback: use a more sophisticated backtracking approach
        # This should rarely be needed given the problem constraints

        # Backtracking with MRV heuristic
        adj_set = [set(neigh) for neigh in adj]
        doms = [set(l) for l in lists]

        for i in range(n):
            if not doms[i]:
                return None

        assignment = [-1] * n
        unassigned = set(range(n))

        def backtrack():
            if not unassigned:
                return True

            # Find MRV vertex
            mrv_v = None
            min_size = 10
            for v in unassigned:
                size = len(doms[v])
                if size < min_size:
                    min_size = size
                    mrv_v = v
                    if size == 0:
                        break

            if min_size == 0:
                return False

            unassigned.remove(mrv_v)

            for c in list(doms[mrv_v]):
                assignment[mrv_v] = c
                doms[mrv_v].discard(c)

                changes = []
                conflict = False
                for u in adj_set[mrv_v]:
                    if u in unassigned:
                        if c in doms[u]:
                            doms[u].discard(c)
                            changes.append(u)
                            if not doms[u]:
                                conflict = True
                                break

                if not conflict:
                    if backtrack():
                        return True

                # Restore
                for u in changes:
                    doms[u].add(c)
                doms[mrv_v].add(c)

            unassigned.add(mrv_v)
            return False

        if backtrack():
            return [int(c) for c in assignment]
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = color_instance(n, edges, lists)
        if coloring is None:
            # Should not happen per problem statement, but handle gracefully
            coloring = [0] * n
        # Ensure types are Python int
        out.append([int(c) for c in coloring])
    return out
# EVOLVE-BLOCK-END
