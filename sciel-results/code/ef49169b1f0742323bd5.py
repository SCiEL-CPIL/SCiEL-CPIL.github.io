"""Model-and-optimize: Min-Conflicts local search for list coloring on bipartite graphs."""
# EVOLVE-BLOCK-START
import time
from random import Random


def construct():
    t0 = time.time()
    LIMIT = 120.0
    MARGIN = 15.0

    def deadline():
        return (time.time() - t0) > (LIMIT - MARGIN)

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
        """
        Solves the list coloring problem using Min-Conflicts local search.
        Optimized with precomputed neighbor color counts for faster delta evaluation.
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        rng = Random(12345)

        # Multiple restarts of Min-Conflicts
        max_restarts = 300
        max_steps_per_restart = 80000

        for restart in range(max_restarts):
            if deadline():
                break

            # Initialize: random assignment from lists
            colour = [rng.choice(lists[v]) for v in range(n)]

            # Precompute conflict counts and neighbor color counts
            conflict_count = [0] * n
            neighbor_color_count = [{} for _ in range(n)]
            for v in range(n):
                cnt = {}
                cv = colour[v]
                for u in adj[v]:
                    c = colour[u]
                    cnt[c] = cnt.get(c, 0) + 1
                neighbor_color_count[v] = cnt
                conflict_count[v] = cnt.get(cv, 0)

            # List violations
            list_violation = [colour[v] not in lists[v] for v in range(n)]
            total_list_viol = sum(list_violation)
            total_edge_conflicts = sum(conflict_count) // 2

            # Check if solution is valid
            if total_edge_conflicts == 0 and total_list_viol == 0:
                return [int(c) for c in colour]

            # Min-Conflicts: repeatedly pick a conflicted vertex and reassign
            for step in range(max_steps_per_restart):
                if deadline():
                    break

                # Find all conflicted vertices
                conflicted = []
                for v in range(n):
                    if list_violation[v] or conflict_count[v] > 0:
                        conflicted.append(v)

                if not conflicted:
                    return [int(c) for c in colour]

                # Pick a random conflicted vertex
                v = rng.choice(conflicted)
                current_colour = colour[v]

                # Find the color from lists[v] that minimizes total conflicts
                best_colour = None
                best_conflict_delta = float('inf')

                # Use precomputed neighbor color counts for fast delta calculation
                ncc = neighbor_color_count[v]
                count_current = ncc.get(current_colour, 0)

                for c in lists[v]:
                    if c == current_colour:
                        continue

                    count_c = ncc.get(c, 0)

                    # Delta in total_conflicts = (count_c - count_current) + list_violation_change
                    delta = count_c - count_current

                    if list_violation[v]:
                        delta -= 1  # c is in lists[v], so list violation is resolved

                    if delta < best_conflict_delta:
                        best_conflict_delta = delta
                        best_colour = c

                if best_colour is None:
                    best_colour = rng.choice(lists[v])

                # Apply the change
                old_colour = current_colour
                colour[v] = best_colour

                # Update v's conflict count
                conflict_count[v] = neighbor_color_count[v].get(best_colour, 0)

                # Update neighbor color counts and conflict counts for v's neighbors
                for u in adj[v]:
                    nc = neighbor_color_count[u]
                    nc[old_colour] = nc.get(old_colour, 0) - 1
                    if nc[old_colour] <= 0:
                        del nc[old_colour]
                    nc[best_colour] = nc.get(best_colour, 0) + 1

                    if colour[u] == old_colour:
                        conflict_count[u] -= 1
                    if colour[u] == best_colour:
                        conflict_count[u] += 1

                # Update list violation
                if list_violation[v]:
                    list_violation[v] = False
                    total_list_viol -= 1

                # Check solution
                if sum(conflict_count) == 0 and total_list_viol == 0:
                    return [int(c) for c in colour]

        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if deadline():
            n, edges, lists = instance(splits, seed)
            rng_fallback = Random(777)
            out.append([int(rng_fallback.choice(lists[v])) for v in range(n)])
            continue

        n, edges, lists = instance(splits, seed)
        colour = solve_instance(n, edges, lists)

        if colour is None:
            # Fallback to random if solver fails
            rng_fallback = Random(777)
            colour = [int(rng_fallback.choice(lists[v])) for v in range(n)]

        out.append(colour)

    return out
# EVOLVE-BLOCK-END
