"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

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

    def solve_instance(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Initialize available colors for each vertex as sets for fast discard/add
        avail = [set(l) for l in lists]
        color = [-1] * n

        # Forward-checking backtracking solver
        # Maintains avail[v] as the set of colors still possible for uncolored v
        # When we assign color c to vertex v, we remove c from avail[u] for each uncolored neighbor u
        # If any uncolored neighbor's avail becomes empty, we backtrack

        def backtrack(remaining):
            if remaining == 0:
                return True

            # MRV: Find uncolored vertex with fewest available colors
            best_v = -1
            best_count = float('inf')
            for v in range(n):
                if color[v] == -1:
                    cnt = len(avail[v])
                    if cnt == 0:
                        return False  # Dead end: no color available
                    if cnt < best_count:
                        best_count = cnt
                        best_v = v
                        if cnt == 1:
                            break  # Can't do better than 1

            # LCV: Sort colors by how many options they remove from uncolored neighbors
            # Try the color that removes the fewest options first
            def lcv_score(c):
                count = 0
                for u in adj[best_v]:
                    if color[u] == -1 and c in avail[u]:
                        count += 1
                return count

            sorted_colors = sorted(list(avail[best_v]), key=lcv_score)

            for c in sorted_colors:
                color[best_v] = c
                # Forward checking: remove c from neighbors' avail sets
                undo_list = []
                valid = True
                for u in adj[best_v]:
                    if color[u] == -1:
                        if c in avail[u]:
                            avail[u].discard(c)
                            undo_list.append((u, c))
                            if len(avail[u]) == 0:
                                valid = False
                                break

                if valid:
                    if backtrack(remaining - 1):
                        return True

                # Undo forward checking
                for u, c_undo in undo_list:
                    avail[u].add(c_undo)
                color[best_v] = -1

            return False

        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(10000)
        success = backtrack(n)
        sys.setrecursionlimit(old_limit)

        if success:
            return [int(c) for c in color]
        else:
            # Fallback: should not happen by Alon-Tarsi theorem
            return [int(l[0]) for l in lists]

    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # If we run out of time, generate a dummy coloring with correct length
            n, edges, lists = instance(splits, seed)
            out.append([int(l[0]) for l in lists])
            continue

        n, edges, lists = instance(splits, seed)
        colouring = solve_instance(n, edges, lists)
        out.append(colouring)

    return out
# EVOLVE-BLOCK-END
