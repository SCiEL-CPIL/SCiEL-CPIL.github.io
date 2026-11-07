"""Solver for large clique subdivisions in a pinned graph."""

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    import time
    from collections import deque
    from itertools import combinations
    import random

    t_start = time.time()
    DEADLINE = 55.0

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    adj_list = [list() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        adj_list[i] = sorted(adj[i])

    def time_up():
        return time.time() - t_start > DEADLINE

    # Precompute all-pairs shortest paths
    sp_dist = [[0] * n for _ in range(n)]
    for s in range(n):
        dist = [float('inf')] * n
        dist[s] = 0
        q = deque([s])
        while q:
            cur = q.popleft()
            for nb in adj_list[cur]:
                if dist[nb] > dist[cur] + 1:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        sp_dist[s] = dist

    def bfs_path(src, dst, forbidden):
        """BFS from src to dst avoiding forbidden interior vertices."""
        if src == dst:
            return [src]
        if dst in forbidden and dst != src:
            return None
        parent = {src: -1}
        q = deque([src])
        while q:
            cur = q.popleft()
            if cur == dst:
                path = []
                node = dst
                while node != -1:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in adj_list[cur]:
                if nb not in parent:
                    if nb != dst and nb in forbidden:
                        continue
                    parent[nb] = cur
                    q.append(nb)
        return None

    def bfs_path_with_length_limit(src, dst, forbidden, max_len):
        """BFS with path length limit."""
        if src == dst:
            return [src]
        if dst in forbidden:
            return None
        parent = {src: -1}
        depth = {src: 0}
        q = deque([src])
        while q:
            cur = q.popleft()
            if depth[cur] >= max_len:
                continue
            if cur == dst:
                path = []
                node = dst
                while node != -1:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in adj_list[cur]:
                if nb not in parent:
                    if nb != dst and nb in forbidden:
                        continue
                    parent[nb] = cur
                    depth[nb] = depth[cur] + 1
                    q.append(nb)
        return None

    def try_assign(branches, order, used_extra=None):
        """Try to assign disjoint paths for all pairs in given order."""
        bset = frozenset(branches)
        used = set()
        if used_extra:
            used |= used_extra
        paths = {}
        for u, v in order:
            path = bfs_path(u, v, used)
            if path is None:
                return None
            interior = set(path[1:-1])
            if interior & bset or interior & used:
                return None
            used |= interior
            paths[(min(u, v), max(u, v))] = path
        return paths

    def try_backtrack(branches, order, max_alts=4):
        """Backtracking: at each step, try multiple path alternatives."""
        bset = frozenset(branches)

        def bt(idx, used, paths):
            if time_up():
                return None
            if idx == len(order):
                return paths

            u, v = order[idx]
            alts = []
            p0 = bfs_path(u, v, used)
            if p0 is None:
                return None
            alts.append(p0)
            # Try excluding each interior vertex of p0 to get alternatives
            for mid in p0[1:-1]:
                if time_up() or len(alts) >= max_alts:
                    break
                alt = bfs_path(u, v, used | {mid})
                if alt is not None and alt not in alts:
                    alts.append(alt)

            for path in alts:
                interior = set(path[1:-1])
                if interior & bset or interior & used:
                    continue
                new_used = used | interior
                paths[(min(u, v), max(u, v))] = path
                result = bt(idx + 1, new_used, paths)
                if result is not None:
                    return result
                del paths[(min(u, v), max(u, v))]
            return None

        return bt(0, set(), {})

    def make_result(branches, paths):
        bset = set(branches)
        path_list = []
        for u, v in combinations(branches, 2):
            key = (min(u, v), max(u, v))
            path = paths[key]
            path_list.append(list(path))
        return {'branches': list(branches), 'paths': path_list}

    # Degree-based ordering
    deg = [len(adj[i]) for i in range(n)]
    candidates = sorted(range(n), key=lambda i: (-deg[i], i))

    best = None
    best_size = 0

    # Strategy 1: Greedy with multiple orderings, try sizes from large to small
    max_size = min(n, 12)
    for size in range(max_size, 1, -1):
        if time_up():
            break
        if size <= best_size:
            break

        top_k = min(n, size + 6)
        pool = candidates[:top_k]
        tried = 0
        max_tries = 8000

        for combo in combinations(pool, size):
            if time_up() or tried >= max_tries:
                break
            tried += 1
            branches = list(combo)

            pairs = list(combinations(branches, 2))
            deg_map = {v: deg[v] for v in branches}

            # Multiple orderings
            order1 = sorted(pairs, key=lambda p: (
                0 if p[1] in adj[p[0]] else 1,
                sp_dist[p[0]][p[1]],
                p[0], p[1]
            ))
            order2 = sorted(pairs, key=lambda p: -(deg_map[p[0]] + deg_map[p[1]]))
            order3 = sorted(pairs, key=lambda p: (deg_map[p[0]], deg_map[p[1]]))
            order4 = list(reversed(order1))
            # Order by product of degrees (constrained pairs first)
            order5 = sorted(pairs, key=lambda p: (deg_map[p[0]] * deg_map[p[1]]))

            found = False
            for order in [order1, order2, order3, order4, order5]:
                if time_up():
                    break
                paths = try_assign(branches, order)
                if paths is not None:
                    best = make_result(branches, paths)
                    best_size = size
                    found = True
                    break

            if found:
                break

        if best_size >= size:
            break

    # Strategy 2: Backtracking on promising sets
    if best is None or best_size < 8:
        for size in range(min(n, 10), max(best_size, 3), -1):
            if time_up():
                break
            if size <= best_size:
                break
            top_k = min(n, size + 5)
            pool = candidates[:top_k]
            tried = 0
            max_tries = 3000
            for combo in combinations(pool, size):
                if time_up() or tried >= max_tries:
                    break
                tried += 1
                branches = list(combo)
                pairs = list(combinations(branches, 2))
                deg_map = {v: deg[v] for v in branches}
                order = sorted(pairs, key=lambda p: (
                    0 if p[1] in adj[p[0]] else 1,
                    sp_dist[p[0]][p[1]],
                    p[0], p[1]
                ))
                paths = try_backtrack(branches, order, max_alts=5)
                if paths is not None:
                    best = make_result(branches, paths)
                    best_size = size
                    break
            if best_size >= size:
                break

    # Strategy 3: Local search - start from a good solution and try to grow
    if best is not None and best_size < 10:
        current_branches = list(best['branches'])
        improved = True
        while improved and not time_up():
            improved = False
            # Try adding each non-branch vertex
            for v in range(n):
                if time_up():
                    break
                if v in current_branches:
                    continue
                new_branches = current_branches + [v]
                new_branches.sort()
                pairs = list(combinations(new_branches, 2))
                deg_map = {x: deg[x] for x in new_branches}
                order = sorted(pairs, key=lambda p: (
                    0 if p[1] in adj[p[0]] else 1,
                    sp_dist[p[0]][p[1]],
                    p[0], p[1]
                ))
                paths = try_backtrack(new_branches, order, max_alts=4)
                if paths is not None:
                    best = make_result(new_branches, paths)
                    best_size = len(new_branches)
                    current_branches = new_branches
                    improved = True
                    break

    # Fallback: find a triangle
    if best is None:
        for a, b, c in combinations(range(n), 3):
            if b in adj[a] and c in adj[a] and c in adj[b]:
                best = {
                    'branches': [a, b, c],
                    'paths': [[a, b], [a, c], [b, c]]
                }
                break
        if best is None:
            u, v = edges[0]
            best = {'branches': [u, v], 'paths': [[u, v]]}

    # Convert to proper types
    branches = [int(x) for x in best['branches']]
    paths = [[int(x) for x in p] for p in best['paths']]
    return {'branches': branches, 'paths': paths}
