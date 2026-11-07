"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

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
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Convert to lists for faster iteration
        adj_list = [list(s) for s in adj]

        # Exact backtracking with MRV (Minimum Remaining Values) and LCV (Least Constraining Value)
        colour = [-1] * n

        import sys
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(10000)

        def backtrack(remaining):
            """
            remaining: number of uncoloured vertices
            Returns True if a valid colouring is found.
            """
            if remaining == 0:
                return True

            # Find the uncoloured vertex with the fewest available colours (MRV)
            best_v = -1
            best_avail = None
            best_count = float('inf')

            # Iterate over uncoloured vertices
            for v in range(n):
                if colour[v] != -1:
                    continue
                # Compute available colours for v
                taken = set()
                for u in adj_list[v]:
                    if colour[u] != -1:
                        taken.add(colour[u])
                avail = [c for c in lists[v] if c not in taken]
                cnt = len(avail)
                if cnt == 0:
                    return False  # Dead end
                if cnt < best_count:
                    best_count = cnt
                    best_v = v
                    best_avail = avail
                    if cnt == 1:
                        break  # Can't do better than 1

            # LCV: Sort available colors by the number of options they remove from neighbors
            # We want to try the color that removes the fewest options first.
            def lcv_score(c):
                count = 0
                for u in adj_list[best_v]:
                    if colour[u] == -1 and c in lists[u]:
                        # This color c is in u's list, so assigning c to best_v removes c from u's options
                        count += 1
                return count

            best_avail.sort(key=lcv_score)

            # Assign colour to best_v
            for c in best_avail:
                colour[best_v] = c
                if backtrack(remaining - 1):
                    return True
            colour[best_v] = -1
            return False

        # Start backtracking
        success = backtrack(n)

        sys.setrecursionlimit(old_limit)

        if success:
            # Ensure all are Python ints
            return [int(c) for c in colour]
        else:
            # Fallback: should not happen by Alon-Tarsi, but return something
            # to avoid crashing. Return list of 0s (will likely fail validation
            # for some edges but at least type-correct).
            return [0] * n

    import time
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # If we run out of time, return what we have (padded with invalid colorings if necessary)
            # But since we solve sequentially, we should have finished most.
            # To be safe, if we haven't solved this one, we return a dummy invalid coloring.
            # However, the task says "wrong length or type makes the whole result invalid".
            # So we must return a list of 4 lists.
            # If we fail, we can't really "return best result" for a binary score unless we have a partial valid one?
            # No, score is 0-4. If one is wrong, it's 0 for that instance.
            # But if we don't return a list of 4, the whole thing is invalid.
            # So we must return something.
            # Given the speed, this is highly unlikely.
            out.append([0] * (8 + 4 * splits))
            continue

        n, edges, lists = instance(splits, seed)
        colouring = solve_instance(n, edges, lists)
        out.append(colouring)

    return out
# EVOLVE-BLOCK-END
