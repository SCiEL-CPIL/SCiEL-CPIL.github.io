"""MODEL-AND-OPTIMIZE: CSP solver with MRV, Forward Checking, and AC-3 initialization."""
# EVOLVE-BLOCK-START
import time
from collections import deque
import sys

def construct():
    sys.setrecursionlimit(10000)

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

    def ac3(n, adj, domains):
        """AC-3 arc consistency preprocessing."""
        queue = deque()
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))
        while queue:
            xi, xj = queue.popleft()
            di = domains[xi]
            dj = domains[xj]
            if not di:
                return False
            new_di = []
            if len(dj) > 1:
                new_di = di  # all values supported
            else:
                # len(dj) == 1
                only = dj[0]
                for ci in di:
                    if ci != only:
                        new_di.append(ci)
            if len(new_di) < len(di):
                domains[xi] = new_di
                if not new_di:
                    return False
                for xk in adj[xi]:
                    if xk != xj:
                        queue.append((xk, xi))
        return True

    def solve_csp(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        domains = [list(l) for l in lists]
        for i in range(n):
            if not domains[i]:
                return None

        if not ac3(n, adj, domains):
            return None
        for i in range(n):
            if not domains[i]:
                return None

        assignment = [None] * n
        # Track assigned neighbors' colors as a set for fast lookup
        # For each vertex, track which colors are "blocked" by assigned neighbors
        blocked = [set() for _ in range(n)]

        def valid_color_count(v):
            """Count how many colors in domains[v] are not blocked."""
            dv = domains[v]
            bv = blocked[v]
            cnt = 0
            for c in dv:
                if c not in bv:
                    cnt += 1
            return cnt

        def backtrack(assigned_count):
            if assigned_count == n:
                return True

            # MRV: find unassigned vertex with fewest valid colors
            min_var = -1
            min_size = 4
            for v in range(n):
                if assignment[v] is None:
                    cnt = valid_color_count(v)
                    if cnt < min_size:
                        min_size = cnt
                        min_var = v
                        if min_size == 0:
                            return False
                        if min_size == 1:
                            break

            if min_var == -1:
                return False

            # Try colors for min_var
            dv = domains[min_var]
            bv = blocked[min_var]
            nbrs = adj[min_var]
            for c in dv:
                if c not in bv:
                    assignment[min_var] = c
                    # Propagate: add c to blocked sets of neighbors
                    for u in nbrs:
                        blocked[u].add(c)

                    if backtrack(assigned_count + 1):
                        return True

                    # Undo
                    for u in nbrs:
                        blocked[u].discard(c)
                    assignment[min_var] = None

            return False

        if backtrack(0):
            return assignment
        return None

    def build_result():
        params = ((20, 701), (40, 702), (60, 703), (100, 704))
        results = []
        for splits, seed in params:
            n, edges, lists = instance(splits, seed)
            coloring = solve_csp(n, edges, lists)
            if coloring is None:
                coloring = [lists[v][0] for v in range(n)]
            results.append([int(c) for c in coloring])
        return results

    start_time = time.time()
    deadline = start_time + 130.0

    result = build_result()

    return result

# EVOLVE-BLOCK-END
