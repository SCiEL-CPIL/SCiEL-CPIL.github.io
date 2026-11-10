import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    best_branches = None
    best_paths = None

    def bfs_path(start, end, forbidden, adj):
        if start == end:
            return [start]
        visited = {start}
        parent = {start: None}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if u == end:
                path = []
                node = end
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for v in adj[u]:
                if v in visited:
                    continue
                if v in forbidden and v != end:
                    continue
                visited.add(v)
                parent[v] = u
                queue.append(v)
        return None

    def try_branch_set(branches):
        if len(branches) < 2:
            return None
        k = len(branches)
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))

        # Try multiple orderings and different path-finding strategies
        import random
        rng = random.Random(42)

        # Strategy 1: Greedy with BFS shortest path
        def find_paths_greedy(remaining_pairs, used_int, current_paths):
            if time.time() > deadline:
                return None
            if not remaining_pairs:
                return current_paths

            # Try each remaining pair
            for idx, (a, b) in enumerate(remaining_pairs):
                rest = remaining_pairs[:idx] + remaining_pairs[idx+1:]
                forbidden = used_int | (branch_set - {a, b})
                path = bfs_path(a, b, forbidden, adj)
                if path is not None:
                    interior = set(path[1:-1])
                    if not (interior & used_int):
                        new_used = used_int | interior
                        result = find_paths_greedy(rest, new_used, current_paths + [path])
                        if result is not None:
                            return result
            return None

        # Strategy 2: Greedy but try to find longer paths when shortest fails
        def bfs_path_length(start, end, forbidden, adj, max_len):
            """Find a path of length <= max_len from start to end avoiding forbidden."""
            if start == end:
                return [start]
            if max_len <= 1:
                if end in adj[start] and end not in forbidden:
                    return [start, end]
                return None

            visited = {start}
            parent = {start: None}
            depth = {start: 0}
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if depth[u] >= max_len:
                    continue
                if u == end:
                    path = []
                    node = end
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                for v in adj[u]:
                    if v in visited:
                        continue
                    if v in forbidden and v != end:
                        continue
                    visited.add(v)
                    parent[v] = u
                    depth[v] = depth[u] + 1
                    queue.append(v)
            return None

        def find_paths_greedy_long(remaining_pairs, used_int, current_paths):
            if time.time() > deadline:
                return None
            if not remaining_pairs:
                return current_paths

            for idx, (a, b) in enumerate(remaining_pairs):
                rest = remaining_pairs[:idx] + remaining_pairs[idx+1:]
                forbidden = used_int | (branch_set - {a, b})
                # Try increasing path lengths
                for max_len in range(2, n + 1):
                    if time.time() > deadline:
                        return None
                    path = bfs_path_length(a, b, forbidden, adj, max_len)
                    if path is not None:
                        interior = set(path[1:-1])
                        if not (interior & used_int):
                            new_used = used_int | interior
                            result = find_paths_greedy_long(rest, new_used, current_paths + [path])
                            if result is not None:
                                return result
                    # If we found any path at this length but couldn't complete, 
                    # try next length (the path might block others)
                    # Actually, let's just try the shortest that works
                    break
            return None

        # Try natural order with greedy
        result = find_paths_greedy(pairs, set(), [])
        if result is not None:
            return result

        # Try sorted by "difficulty" (shorter available paths first)
        def pair_difficulty(pair):
            a, b = pair
            forbidden = branch_set - {a, b}
            path = bfs_path(a, b, forbidden, adj)
            return len(path) if path else 999

        sorted_pairs = sorted(pairs, key=pair_difficulty)
        result = find_paths_greedy(sorted_pairs, set(), [])
        if result is not None:
            return result

        # Try randomized orderings
        for attempt in range(10):
            if time.time() > deadline:
                return None
            shuffled = pairs[:]
            rng.shuffle(shuffled)
            result = find_paths_greedy(shuffled, set(), [])
            if result is not None:
                return result

        # Try with longer path search
        for attempt in range(5):
            if time.time() > deadline:
                return None
            shuffled = pairs[:]
            rng.shuffle(shuffled)
            result = find_paths_greedy_long(shuffled, set(), [])
            if result is not None:
                return result

        return None

    # Find initial clique
    def find_max_clique_greedy(adj, n):
        best = []
        for start in range(n):
            if time.time() > deadline:
                break
            clique = [start]
            candidates = set(adj[start])
            while candidates:
                best_v = max(candidates, key=lambda v: len(adj[v] & set(clique)))
                if len(adj[best_v] & set(clique)) < len(clique):
                    break
                clique.append(best_v)
                candidates = candidates & adj[best_v]
                candidates = candidates - set(clique)
            if len(clique) > len(best):
                best = clique[:]
        return best

    initial_clique = find_max_clique_greedy(adj, n)
    paths_result = try_branch_set(initial_clique)
    if paths_result is not None:
        best_branches = initial_clique[:]
        best_paths = paths_result

    # Greedy expansion
    if best_branches is not None:
        current_branches = best_branches[:]
        expanded = True
        while expanded and time.time() < deadline:
            expanded = False
            current_set = set(current_branches)
            candidates = [v for v in range(n) if v not in current_set]
            candidates.sort(key=lambda v: len(adj[v]), reverse=True)

            for v in candidates:
                if time.time() > deadline:
                    break
                new_branches = current_branches + [v]
                paths_result = try_branch_set(new_branches)
                if paths_result is not None:
                    current_branches = new_branches
                    best_branches = current_branches[:]
                    best_paths = paths_result
                    expanded = True
                    break

    # Random search for larger sets
    import random
    rng2 = random.Random(123)

    for size in range(n, 1, -1):
        if time.time() > deadline:
            break
        if best_branches is not None and size <= len(best_branches):
            break
        for attempt in range(200):
            if time.time() > deadline:
                break
            random_set = rng2.sample(range(n), size)
            paths_result = try_branch_set(random_set)
            if paths_result is not None:
                best_branches = random_set[:]
                best_paths = paths_result
                break

    # Local search: try swapping vertices in the best set
    if best_branches is not None and time.time() < deadline:
        current_branches = best_branches[:]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_set = set(current_branches)
            for i in range(len(current_branches)):
                if time.time() > deadline:
                    break
                for v in range(n):
                    if v in current_set:
                        continue
                    # Try swapping current_branches[i] with v
                    new_branches = current_branches[:i] + [v] + current_branches[i+1:]
                    paths_result = try_branch_set(new_branches)
                    if paths_result is not None:
                        current_branches = new_branches
                        best_branches = current_branches[:]
                        best_paths = paths_result
                        improved = True
                        break
                if improved:
                    break

    if best_branches is None or len(best_branches) < 2:
        u, v = P['edges'][0]
        best_branches = [int(u), int(v)]
        best_paths = [[int(u), int(v)]]
    else:
        best_branches = [int(v) for v in best_branches]
        best_paths = [[int(v) for v in path] for path in best_paths]

    return {'branches': best_branches, 'paths': best_paths}
