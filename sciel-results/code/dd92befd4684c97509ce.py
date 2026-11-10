"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

    start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

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

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def solve_list_coloring(n, adj, lists):
        # Order vertices by degree descending (most constrained first)
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        colour = [None] * n

        # Backtracking with forward checking
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            taken = set()
            for u in adj[v]:
                c = colour[u]
                if c is not None:
                    taken.add(c)
            for c in lists[v]:
                if c not in taken:
                    colour[v] = c
                    if backtrack(idx + 1):
                        return True
                    colour[v] = None
            return False

        # Try simple greedy first (fast path)
        for v in order:
            taken = {colour[u] for u in adj[v] if colour[u] is not None}
            free = [c for c in lists[v] if c not in taken]
            if free:
                colour[v] = free[0]
            else:
                colour[v] = lists[v][0]
        # Verify greedy result
        ok = True
        for u, v in adj_pairs:
            if colour[u] is None or colour[v] is None:
                ok = False
                break
            if colour[u] not in lists[u] or colour[v] not in lists[v]:
                ok = False
                break
            if colour[u] == colour[v]:
                ok = False
                break
        if ok:
            return colour

        # Full backtracking
        colour = [None] * n
        if backtrack(0):
            return colour
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > DEADLINE:
            # Return what we have, padded with None if needed (but this shouldn't happen)
            while len(out) < 4:
                out.append([])
            break
        n, edges, lists = instance(splits, seed)
        adj = build_adj(n, edges)
        adj_pairs = [(u, v) for u in range(n) for v in adj[u] if u < v]

        # Greedy first
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        colour = [None] * n
        greedy_ok = True
        for v in order:
            taken = set()
            for u in adj[v]:
                if colour[u] is not None:
                    taken.add(colour[u])
            free = [c for c in lists[v] if c not in taken]
            if free:
                colour[v] = free[0]
            else:
                colour[v] = lists[v][0]
                greedy_ok = False

        # Verify
        if greedy_ok:
            for u, v in adj_pairs:
                if colour[u] == colour[v]:
                    greedy_ok = False
                    break
                if colour[u] not in lists[u] or colour[v] not in lists[v]:
                    greedy_ok = False
                    break

        if not greedy_ok:
            # Backtracking search
            colour = [None] * n

            def bt(idx):
                if time.time() > DEADLINE - 2.0:
                    return False
                if idx == n:
                    return True
                v = order[idx]
                taken = set()
                for u in adj[v]:
                    c = colour[u]
                    if c is not None:
                        taken.add(c)
                for c in lists[v]:
                    if c not in taken:
                        colour[v] = c
                        if bt(idx + 1):
                            return True
                        colour[v] = None
                return False

            if not bt(0):
                # Fallback: just use greedy assignment (may be invalid but better than nothing)
                colour = [None] * n
                for v in order:
                    taken = set()
                    for u in adj[v]:
                        if colour[u] is not None:
                            taken.add(colour[u])
                    free = [c for c in lists[v] if c not in taken]
                    colour[v] = free[0] if free else lists[v][0]

        # Ensure all are Python ints
        out.append([int(c) for c in colour])

    # Ensure exactly 4 entries
    while len(out) < 4:
        out.append([])
    return out
# EVOLVE-BLOCK-END
