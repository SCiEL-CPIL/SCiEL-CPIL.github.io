"""Solver for large clique subdivisions in a pinned graph."""

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    import time
    from collections import deque
    from itertools import combinations

    t_start = time.time()
    DEADLINE = 80.0

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

    def find_shortest_path(src, dst, forbidden_interior):
        """BFS shortest path from src to dst. Interior vertices must avoid forbidden_interior.
        Endpoints src and dst are always allowed."""
        if src == dst:
            return [src]
        parent = {src: -1}
        queue = deque([src])
        while queue:
            cur = queue.popleft()
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
                    if nb != dst and nb in forbidden_interior:
                        continue
                    parent[nb] = cur
                    queue.append(nb)
        return None

    def find_path_alternatives(src, dst, forbidden_interior, max_alt=3):
        """Find up to max_alt distinct shortest paths (or near-shortest)."""
        # First find the shortest path length
        sp = find_shortest_path(src, dst, forbidden_interior)
        if sp is None:
            return []
        paths = [sp]
        # Try to find alternative paths by excluding one interior vertex at a time
        if len(sp) > 2:
            for mid in sp[1:-1]:
                if time_up():
                    break
                alt_forbidden = forbidden_interior | {mid}
                alt = find_shortest_path(src, dst, alt_forbidden)
                if alt is not None and alt not in paths:
                    paths.append(alt)
                    if len(paths) >= max_alt:
                        break
        return paths

    def try_greedy(branches, ordering_key=None):
        """Greedy: assign paths in a specific order, using shortest paths."""
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        if ordering_key:
            pairs.sort(key=ordering_key)

        paths = {}
        used_interiors = set()

        for u, v in pairs:
            path = find_shortest_path(u, v, used_interiors)
            if path is None:
                return None
            interior = set(path[1:-1])
            if interior & bset:
                return None
            if interior & used_interiors:
                return None
            used_interiors |= interior
            paths[(u, v)] = path

        path_list = []
        for u, v in combinations(branches, 2):
            path_list.append(paths[(u, v)])
        return {'branches': list(branches), 'paths': path_list}

    def try_greedy_multi_order(branches):
        """Try multiple orderings of pair assignment."""
        bset = set(branches)
        deg = {v: len(adj[v]) for v in branches}
        pairs = list(combinations(branches, 2))

        orderings = []
        # Direct edges first
        orderings.append(sorted(pairs, key=lambda p: (0 if p[1] in adj[p[0]] else 1, p[0], p[1])))
        # High degree sum first
        orderings.append(sorted(pairs, key=lambda p: -(deg[p[0]] + deg[p[1]])))
        # Low degree first (constrained pairs)
        orderings.append(sorted(pairs, key=lambda p: (deg[p[0]], deg[p[1]])))
        # Reverse of direct
        orderings.append(list(reversed(orderings[0])))

        for ordering in orderings:
            if time_up():
                break
            paths = {}
            used_interiors = set()
            success = True
            for u, v in ordering:
                path = find_shortest_path(u, v, used_interiors)
                if path is None:
                    success = False
                    break
                interior = set(path[1:-1])
                if interior & bset or interior & used_interiors:
                    success = False
                    break
                used_interiors |= interior
                paths[(u, v)] = path
            if success:
                path_list = [paths[(u, v)] for u, v in combinations(branches, 2)]
                return {'branches': list(branches), 'paths': path_list}
        return None

    def try_backtrack(branches, max_depth=4):
        """Backtracking with limited depth. Try alternatives for failing pairs."""
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        # Sort: direct edges first
        deg = {v: len(adj[v]) for v in branches}
        pairs.sort(key=lambda p: (0 if p[1] in adj[p[0]] else 1, p[0], p[1]))

        def backtrack(idx, used_interiors, paths):
            if time_up():
                return None
            if idx == len(pairs):
                path_list = [paths[(u, v)] for u, v in combinations(branches, 2)]
                return {'branches': list(branches), 'paths': path_list}

            u, v = pairs[idx]
            # Find alternatives
            alts = find_path_alternatives(u, v, used_interiors, max_alt=4)
            for path in alts:
                interior = set(path[1:-1])
                if interior & bset or interior & used_interiors:
                    continue
                new_used = used_interiors | interior
                paths[(u, v)] = path
                result = backtrack(idx + 1, new_used, paths)
                if result is not None:
                    return result
                del paths[(u, v)]
            return None

        return backtrack(0, set(), {})

    # Compute degree and other metrics
    deg = [len(adj[i]) for i in range(n)]

    # Score candidates: high degree, well-connected to other high-degree vertices
    def candidate_score(v):
        return deg[v]

    candidates = sorted(range(n), key=lambda i: (-deg[i], i))

    best = None
    best_size = 0

    # Try sizes from large to small
    for size in range(min(n, 14), 1, -1):
        if time_up():
            break
        if size <= best_size:
            break

        # Use top candidates for this size
        top_k = min(n, size + 5)
        tried = 0
        max_tries = 2000

        for combo in combinations(candidates[:top_k], size):
            if time_up() or tried >= max_tries:
                break
            tried += 1
            branches = list(combo)

            # First try greedy with multiple orderings (fast)
            result = try_greedy_multi_order(branches)
            if result is None and not time_up():
                # Try backtracking (slower but more thorough)
                result = try_backtrack(branches)

            if result is not None:
                if size > best_size:
                    best = result
                    best_size = size
                break  # Found one of this size, move to larger

        if best_size >= size:
            break

    # Fallback: try a few more with backtracking if we didn't find large
    if best is None or best_size < 8:
        for size in range(min(n, 10), max(best_size, 3), -1):
            if time_up():
                break
            if size <= best_size:
                break
            top_k = min(n, size + 4)
            tried = 0
            max_tries = 1000
            for combo in combinations(candidates[:top_k], size):
                if time_up() or tried >= max_tries:
                    break
                tried += 1
                branches = list(combo)
                result = try_backtrack(branches)
                if result is not None:
                    if size > best_size:
                        best = result
                        best_size = size
                    break
            if best_size >= size:
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
