"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct():
    from random import Random
    import sys

    start_time = time.time()
    DEADLINE = 145.0

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

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

    def find_bipartition(n, adj):
        color = [-1] * n
        side_a = set()
        side_b = set()
        for start in range(n):
            if color[start] != -1:
                continue
            color[start] = 0
            side_a.add(start)
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        if color[v] == 0:
                            side_a.add(v)
                        else:
                            side_b.add(v)
                        q.append(v)
        return side_a, side_b

    def solve_instance(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        side_a, side_b = find_bipartition(n, adj)

        # Choose smaller side to color first
        if len(side_a) <= len(side_b):
            first_side = list(side_a)
            second_side = list(side_b)
        else:
            first_side = list(side_b)
            second_side = list(side_a)

        # Sort by degree descending (most constrained first)
        first_side.sort(key=lambda v: -len(adj[v]))
        second_side.sort(key=lambda v: -len(adj[v]))

        colour = [None] * n
        domains = [frozenset(lst) for lst in lists]

        # Precompute for each vertex in first_side, which vertices in second_side are neighbors
        first_to_second = {}
        for v in first_side:
            first_to_second[v] = [u for u in adj[v] if u in side_b or (side_b == set(second_side) and u in set(second_side))]

        # For faster lookup, build a set for second_side
        second_set = set(second_side)
        first_to_second = {}
        for v in first_side:
            first_to_second[v] = [u for u in adj[v] if u in second_set]

        sys.setrecursionlimit(10000)

        # Phase 1: Backtrack on first side with optimized forward checking
        def backtrack_first(idx):
            if idx == len(first_side):
                return True
            if time_remaining() < 0.05:
                return False

            v = first_side[idx]
            used = set()
            for u in adj[v]:
                cu = colour[u]
                if cu is not None:
                    used.add(cu)

            avail = domains[v] - used
            if not avail:
                return False

            for c in sorted(avail):
                colour[v] = c
                # Forward checking: only check second_side neighbors of v
                ok = True
                for u in first_to_second[v]:
                    if colour[u] is not None:
                        continue
                    used_u = set()
                    for w in adj[u]:
                        cw = colour[w]
                        if cw is not None:
                            used_u.add(cw)
                    if not (domains[u] - used_u):
                        ok = False
                        break
                if ok:
                    if backtrack_first(idx + 1):
                        return True
                colour[v] = None
            return False

        # Phase 2: Greedy color second side
        def color_second():
            for v in second_side:
                used = set()
                for u in adj[v]:
                    cu = colour[u]
                    if cu is not None:
                        used.add(cu)
                available = domains[v] - used
                if not available:
                    return False
                colour[v] = min(available)
            return True

        success = False
        if backtrack_first(0):
            if color_second():
                success = True

        if not success:
            # Fallback: full backtracking with MRV ordering
            colour = [None] * n
            order = sorted(range(n), key=lambda u: -len(adj[u]))
            pos = {v: i for i, v in enumerate(order)}
            later = [[] for _ in range(n)]
            for v in range(n):
                pv = pos[v]
                for u in adj[v]:
                    if pos[u] > pv:
                        later[v].append(u)

            def backtrack_full(idx):
                if idx == n:
                    return True
                if time_remaining() < 0.05:
                    return False
                v = order[idx]
                used = set()
                for u in adj[v]:
                    cu = colour[u]
                    if cu is not None:
                        used.add(cu)
                avail = domains[v] - used
                if not avail:
                    return False
                for c in sorted(avail):
                    colour[v] = c
                    ok = True
                    for u in later[v]:
                        if colour[u] is not None:
                            continue
                        used_u = set()
                        for w in adj[u]:
                            cw = colour[w]
                            if cw is not None:
                                used_u.add(cw)
                        if not (domains[u] - used_u):
                            ok = False
                            break
                    if ok:
                        if backtrack_full(idx + 1):
                            return True
                    colour[v] = None
                return False

            if not backtrack_full(0):
                # Last resort: greedy fill
                for v in range(n):
                    if colour[v] is None:
                        colour[v] = lists[v][0]

        return colour

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time_remaining() < 1.0:
            # Not enough time; generate minimal valid-length output
            n, edges, lists = instance(splits, seed)
            out.append([lists[v][0] for v in range(n)])
            continue
        n, edges, lists = instance(splits, seed)
        out.append(solve_instance(n, edges, lists))

    final_out = []
    for lst in out:
        if isinstance(lst, (list, tuple)):
            final_out.append([int(x) for x in lst])
        else:
            final_out.append(lst)

    return final_out
# EVOLVE-BLOCK-END
