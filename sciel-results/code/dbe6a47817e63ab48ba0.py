"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    # Deadline guard: 150s limit, aim to finish in < 10s
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

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

        # Initial domains as lists (for faster membership and iteration)
        domains = [list(l) for l in lists]

        # Check for empty domains initially
        for i in range(n):
            if not domains[i]:
                return [0] * n

        # Determine bipartition of the graph
        # BFS from vertex 0
        color_part = [-1] * n  # 0 or 1 for bipartition
        from collections import deque
        q = deque([0])
        color_part[0] = 0
        while q:
            u = q.popleft()
            for v in adj[u]:
                if color_part[v] == -1:
                    color_part[v] = 1 - color_part[u]
                    q.append(v)
                elif color_part[v] == color_part[u]:
                    # Not bipartite? Shouldn't happen for quadrangulations
                    pass

        # Partition vertices
        part0 = [i for i in range(n) if color_part[i] == 0]
        part1 = [i for i in range(n) if color_part[i] == 1]

        # Arc Consistency (AC-3) Preprocessing
        def enforce_arc_consistency():
            queue = []
            for u in range(n):
                for v in adj[u]:
                    queue.append((u, v))

            while queue:
                u, v = queue.pop()
                du = set(domains[u])
                dv = set(domains[v])

                if not dv:
                    return False

                # Values in du that have no support in dv
                to_remove = set()
                for c in du:
                    if dv == {c}:
                        to_remove.add(c)

                if to_remove:
                    du.difference_update(to_remove)
                    if not du:
                        return False
                    domains[u] = list(du)
                    for w in adj[u]:
                        if w != v:
                            queue.append((w, u))
            return True

        if not enforce_arc_consistency():
            return [0] * n

        # Use backtracking with MRV + Forward Checking
        sys.setrecursionlimit(10000)

        # Work with sets for domain operations
        cur_domains = [set(d) for d in domains]
        colour = [-1] * n
        unassigned = set(range(n))
        degrees = [len(adj[i]) for i in range(n)]

        def backtrack():
            if time.time() > deadline:
                return False

            if not unassigned:
                return True

            # MRV: pick vertex with smallest domain
            best_v = -1
            best_count = float('inf')
            best_deg = -1

            for v in unassigned:
                cnt = len(cur_domains[v])
                if cnt == 0:
                    return False
                if cnt < best_count or (cnt == best_count and degrees[v] > best_deg):
                    best_count = cnt
                    best_v = v
                    best_deg = degrees[v]
                    if cnt == 1:
                        break

            if best_v == -1:
                return False

            v = best_v
            candidates = list(cur_domains[v])

            # LCV: least constraining value first
            def lcv(c):
                s = 0
                for u in adj[v]:
                    if u in unassigned and c in cur_domains[u]:
                        s += 1
                return s

            candidates.sort(key=lcv)

            for c in candidates:
                colour[v] = c
                unassigned.remove(v)

                saved = []
                conflict = False
                for u in adj[v]:
                    if u in unassigned and c in cur_domains[u]:
                        saved.append(u)
                        cur_domains[u].discard(c)
                        if not cur_domains[u]:
                            conflict = True
                            break

                if not conflict and backtrack():
                    return True

                # Undo
                colour[v] = -1
                unassigned.add(v)
                for u in saved:
                    cur_domains[u].add(c)

            return False

        if backtrack():
            return [int(c) for c in colour]
        else:
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline - 1.0:
            break
        n, edges, lists = instance(splits, seed)
        colouring = solve_instance(n, edges, lists)
        out.append(colouring)

    # Ensure we return exactly 4 lists
    while len(out) < 4:
        # Generate the missing instance to get correct length
        remaining = [(20, 701), (40, 702), (60, 703), (100, 704)]
        idx = len(out)
        splits, seed = remaining[idx]
        n, edges, lists = instance(splits, seed)
        out.append([0] * n)

    return out
# EVOLVE-BLOCK-END
