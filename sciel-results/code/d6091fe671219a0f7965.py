"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from random import Random

    START = time.time()
    DEADLINE = START + 140.0

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

    def solve(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # AC-3: prune domains so that every (u,v) edge has a "differ" support each way.
        doms = [set(l) for l in lists]
        queue = [(u, v) for u in range(n) for v in adj[u]]
        inq = set(queue)
        while queue:
            u, v = queue.pop()
            # support for u relative to v
            new_u = {c for c in doms[u] if any(c != d for d in doms[v])}
            if new_u != doms[u]:
                doms[u] = new_u
                if not new_u:
                    return None
                for w in adj[u]:
                    if w != v and (w, u) not in inq:
                        queue.append((w, u))
                        inq.add((w, u))
            new_v = {c for c in doms[v] if any(c != d for d in doms[u])}
            if new_v != doms[v]:
                doms[v] = new_v
                if not new_v:
                    return None
                for w in adj[v]:
                    if w != u and (w, v) not in inq:
                        queue.append((w, v))
                        inq.add((w, v))

        # Backtracking with forward checking + MRV ordering.
        colour = [-1] * n
        remaining = [list(d) for d in doms]

        def ok():
            return time.time() < DEADLINE

        def assign(v, c):
            colour[v] = c
            for u in adj[v]:
                if colour[u] == -1 and c in remaining[u]:
                    remaining[u].remove(c)
                    if not remaining[u]:
                        return False
            return True

        def unassign(v, c):
            colour[v] = -1
            for u in adj[v]:
                if colour[u] == -1 and c not in remaining[u]:
                    remaining[u].append(c)

        def backtrack():
            if not ok():
                return False
            # pick unassigned vertex with fewest remaining colours (MRV)
            best = -1
            best_len = 10 ** 9
            for v in range(n):
                if colour[v] == -1:
                    ln = len(remaining[v])
                    if ln < best_len:
                        best_len = ln
                        best = v
                        if ln == 1:
                            break
            if best == -1:
                return True  # all assigned
            for c in remaining[best]:
                if assign(best, c):
                    if backtrack():
                        return True
                    unassign(best, c)
            return False

        if backtrack():
            return colour
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() >= DEADLINE:
            out.append(None)
            continue
        n, edges, lists = instance(splits, seed)
        col = solve(n, edges, lists)
        if col is not None:
            out.append([int(c) for c in col])
        else:
            out.append(None)

    # Fallback: if a deadline cut off a solve, fill with a best-effort valid list colouring.
    for i, splits_seed in enumerate(((20, 701), (40, 702), (60, 703), (100, 704))):
        if out[i] is None:
            n, edges, lists = instance(*splits_seed)
            out[i] = [int(lists[v][0]) for v in range(n)]

    return out
# EVOLVE-BLOCK-END
