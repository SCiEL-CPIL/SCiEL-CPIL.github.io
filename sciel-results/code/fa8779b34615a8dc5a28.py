"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque
import random

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

DEADLINE = time.time() + 85.0

def construct():
    t0 = time.time()
    n = P['n']
    edges = P['edges']
    edge_set = set()
    for e in edges:
        a, b = e
        edge_set.add((a, b))
        edge_set.add((b, a))
    adj = [set() for _ in range(n)]
    for e in edges:
        a, b = e
        adj[a].add(b)
        adj[b].add(a)

    def time_up():
        return time.time() - t0 > DEADLINE - 3.0

    def fallback():
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    def bfs_path(s, t, blocked):
        """BFS shortest path from s to t avoiding blocked vertices (except s, t)."""
        if s == t:
            return [s]
        prev = {s: None}
        q = deque([s])
        while q:
            cur = q.popleft()
            if cur == t:
                break
            for nb in adj[cur]:
                if nb in prev:
                    continue
                if nb in blocked and nb != s and nb != t:
                    continue
                prev[nb] = cur
                q.append(nb)
        if t not in prev:
            return None
        path = []
        cur = t
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    def try_solve(branches, pair_order):
        """Try to find disjoint interior paths for all pairs in pair_order."""
        used = set()
        all_b = set(branches)
        paths = []
        for (u, v) in pair_order:
            blocked = used | (all_b - {u, v})
            path = bfs_path(u, v, blocked)
            if path is None:
                return None
            interior = set(path[1:-1])
            if interior & used:
                return None
            used |= interior
            paths.append(path)
        return paths

    def order_pairs_by_shortest(branches):
        """Order pairs by their shortest path length ignoring other pairs' constraints."""
        bs = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        scored = []
        for (u, v) in pairs:
            blocked = bs - {u, v}
            # Quick BFS to get length
            if u == v:
                L = 0
            else:
                prev = {u: None}
                q = deque([u])
                dist = {u: 0}
                found = False
                while q:
                    cur = q.popleft()
                    if cur == v:
                        found = True
                        break
                    for nb in adj[cur]:
                        if nb in prev:
                            continue
                        if nb in blocked:
                            continue
                        prev[nb] = cur
                        dist[nb] = dist[cur] + 1
                        q.append(nb)
                L = dist.get(v, 10**9) if found else 10**9
            scored.append((L, (u, v)))
        scored.sort(key=lambda x: x[0])
        return [p for (_, p) in scored]

    def try_solve_with_order(branches):
        order = order_pairs_by_shortest(branches)
        return try_solve(branches, order)

    def try_solve_with_reorder(branches):
        """Try multiple orderings."""
        bs = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        # Try natural order
        r = try_solve(branches, pairs)
        if r is not None:
            return r
        # Try sorted by shortest
        order = order_pairs_by_shortest(branches)
        r = try_solve(branches, order)
        if r is not None:
            return r
        # Try reverse
        r = try_solve(branches, list(reversed(order)))
        if r is not None:
            return r
        return None

    def is_clique(verts):
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                if (verts[i], verts[j]) not in edge_set:
                    return False
        return True

    best = None
    best_size = 0

    # Start with a 2-branch solution
    u, v = edges[0]
    paths = try_solve([u, v], [(u, v)])
    if paths is not None:
        best = {'branches': [int(u), int(v)], 'paths': [[int(x) for x in pp] for pp in paths]}
        best_size = 2

    # Find all triangles
    vertices = list(range(n))
    triangles = []
    for tri in combinations(vertices, 3):
        a, b, c = tri
        if (a, b) in edge_set and (b, c) in edge_set and (a, c) in edge_set:
            triangles.append(tri)

    # Try each triangle
    for tri in triangles:
        if time_up():
            break
        paths = try_solve_with_reorder(list(tri))
        if paths is not None:
            if best_size < 3:
                best = {'branches': [int(x) for x in tri], 'paths': [[int(x) for x in pp] for pp in paths]}
                best_size = 3

    # Greedy expansion from best known solution
    if best_size >= 3:
        base = sorted(best['branches'])
    elif best_size == 2:
        base = sorted(best['branches'])
    else:
        base = [u, v]

    # Try to expand: add one vertex at a time
    improved = True
    max_expansions = 10
    exp_count = 0
    while improved and not time_up() and exp_count < max_expansions:
        improved = False
        exp_count += 1
        current = sorted(base)
        non_branch = [x for x in vertices if x not in set(current)]
        # Sort candidates by degree (higher degree first as heuristic)
        non_branch.sort(key=lambda x: len(adj[x]), reverse=True)
        best_add = None
        for vb in non_branch:
            if time_up():
                break
            new_bs = current + [vb]
            paths = try_solve_with_reorder(new_bs)
            if paths is not None:
                best_add = (vb, paths)
                break
        if best_add is not None:
            vb, paths = best_add
            base = sorted(base + [vb])
            best = {'branches': [int(x) for x in base], 'paths': [[int(x) for x in pp] for pp in paths]}
            best_size = len(base)
            improved = True

    # Randomized search for larger branch sets
    if not time_up():
        random.seed(123)
        max_tries = 2000
        for trial in range(max_tries):
            if time_up():
                break
            # Try sizes from best_size+1 up to 8
            min_size = best_size + 1
            max_size = min(8, n)
            if min_size > max_size:
                break
            size = random.randint(min_size, max_size)
            # Bias towards vertices with high degree
            weights = [len(adj[x]) + 1 for x in vertices]
            subset = random.sample(vertices, size)
            paths = try_solve_with_reorder(subset)
            if paths is not None:
                if len(subset) > best_size:
                    best = {'branches': [int(x) for x in subset], 'paths': [[int(x) for x in pp] for pp in paths]}
                    best_size = len(subset)

    # Additional: try to expand from multiple starting points
    if not time_up() and best_size < 5:
        # Try expanding from different triangles
        for tri in triangles[:20]:
            if time_up():
                break
            if len(tri) <= best_size:
                continue
            base_tri = sorted(tri)
            # Try to expand this triangle
            current = base_tri
            for _ in range(5):
                if time_up():
                    break
                non_branch = [x for x in vertices if x not in set(current)]
                non_branch.sort(key=lambda x: len(adj[x]), reverse=True)
                added = False
                for vb in non_branch:
                    if time_up():
                        break
                    new_bs = current + [vb]
                    paths = try_solve_with_reorder(new_bs)
                    if paths is not None:
                        current = sorted(current + [vb])
                        if len(current) > best_size:
                            best = {'branches': [int(x) for x in current], 'paths': [[int(x) for x in pp] for pp in paths]}
                            best_size = len(current)
                        added = True
                        break
                if not added:
                    break

    if best is None:
        return fallback()

    return best
# EVOLVE-BLOCK-END
