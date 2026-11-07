import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    deadline = time.time() + 55.0
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    def find_all_shortest_paths(src, dst, forbidden):
        """Find all shortest paths from src to dst avoiding forbidden vertices."""
        if src == dst:
            return []
        if dst in forbidden:
            return []
        # BFS to find shortest path length
        dist = {src: 0}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            if node == dst:
                break
            for nb in adj[node]:
                if nb not in dist and nb not in forbidden:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
        if dst not in dist:
            return []
        # DFS to find all shortest paths
        target_len = dist[dst]
        paths = []
        def dfs(node, path):
            if len(paths) > 20:  # Limit paths to explore
                return
            if node == dst:
                paths.append(list(path))
                return
            if dist.get(node, -1) >= target_len:
                return
            for nb in sorted(adj[node]):
                if nb in forbidden:
                    continue
                if dist.get(nb, -1) == dist[node] + 1:
                    path.append(nb)
                    dfs(nb, path)
                    path.pop()
        dfs(src, [src])
        return paths

    def find_bfs_path(src, dst, forbidden):
        """Find a single shortest path from src to dst avoiding forbidden vertices."""
        if src == dst:
            return None
        if dst in forbidden:
            return None
        queue = deque()
        queue.append((src, [src]))
        visited = {src}
        while queue:
            node, path = queue.popleft()
            for nb in sorted(adj[node]):
                if nb == dst:
                    return path + [dst]
                if nb not in forbidden and nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return None

    def backtrack_paths(branches, time_budget=2.0):
        """Use backtracking to find internally disjoint paths for all pairs."""
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        paths = {}
        start_time = time.time()

        def backtrack(idx, used_interior):
            if time.time() - start_time > time_budget:
                return False
            if idx == len(pairs):
                return True
            u, v = pairs[idx]
            forbidden = branch_set - {u, v} | used_interior
            # Get all shortest paths
            all_paths = find_all_shortest_paths(u, v, forbidden)
            if not all_paths:
                return False
            for path in all_paths:
                interior = set(path[1:-1])
                if interior & used_interior:
                    continue
                if interior & branch_set:
                    continue
                paths[(u, v)] = path
                if backtrack(idx + 1, used_interior | interior):
                    return True
                del paths[(u, v)]
            return False

        if backtrack(0, set()):
            return paths
        return None

    def greedy_paths(branches):
        """Greedy path finding - fast but may miss solutions."""
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        paths = {}
        used_interior = set()
        for u, v in pairs:
            forbidden = branch_set - {u, v} | used_interior
            path = find_bfs_path(u, v, forbidden)
            if path is None:
                return None
            for vertex in path[1:-1]:
                used_interior.add(vertex)
            paths[(u, v)] = path
        return paths

    # Find maximum clique using Bron-Kerbosch
    best_clique = []

    def bron_kerbosch(R, P_set, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        pivot_candidates = P_set | X
        if not pivot_candidates:
            return
        pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P_set))
        candidates = [v for v in P_set if v not in adj[pivot]]
        for v in candidates:
            new_R = R | {v}
            new_P = P_set & adj[v]
            new_X = X & adj[v]
            bron_kerbosch(new_R, new_P, new_X)
            P_set = P_set - {v}
            X = X | {v}

    bron_kerbosch(set(), set(range(n)), set())

    best_result = None
    best_count = 0

    # Try the maximum clique with backtracking
    if len(best_clique) >= 2:
        branches = sorted(best_clique)
        paths = backtrack_paths(branches, time_budget=3.0)
        if paths is not None:
            best_count = len(branches)
            result_paths = []
            for u, v in combinations(branches, 2):
                result_paths.append([int(x) for x in paths[(u, v)]])
            best_result = {'branches': [int(x) for x in branches], 'paths': result_paths}

    # Try to expand beyond the clique using greedy + backtracking
    if len(best_clique) >= 2:
        current_branches = list(best_clique)
        for _ in range(5):
            if time.time() > deadline:
                break
            expanded = False
            for candidate in range(n):
                if candidate in current_branches:
                    continue
                if time.time() > deadline:
                    break
                new_branches = current_branches + [candidate]
                # First try fast greedy
                paths = greedy_paths(new_branches)
                if paths is not None:
                    current_branches = new_branches
                    expanded = True
                    break
            if not expanded:
                break

        if len(current_branches) > best_count:
            paths = backtrack_paths(current_branches, time_budget=3.0)
            if paths is not None:
                best_count = len(current_branches)
                result_paths = []
                for u, v in combinations(current_branches, 2):
                    result_paths.append([int(x) for x in paths[(u, v)]])
                best_result = {'branches': [int(x) for x in sorted(current_branches)], 'paths': result_paths}

    # Try random subsets with backtracking - can find non-clique solutions
    import random
    random.seed(42)
    for trial in range(500):
        if time.time() > deadline:
            break
        size = random.randint(3, min(n, 7))
        subset = random.sample(range(n), size)
        if size <= best_count:
            continue
        # Quick greedy check first
        paths = greedy_paths(subset)
        if paths is None:
            # Try backtracking with small budget
            paths = backtrack_paths(subset, time_budget=0.5)
        if paths is not None and len(subset) > best_count:
            best_count = len(subset)
            result_paths = []
            for u, v in combinations(subset, 2):
                result_paths.append([int(x) for x in paths[(u, v)]])
            best_result = {'branches': [int(x) for x in sorted(subset)], 'paths': result_paths}

    # Try systematic expansion from high-degree vertices
    if time.time() < deadline - 10:
        # Start from top-degree vertices
        deg_sorted = sorted(range(n), key=lambda v: -len(adj[v]))
        for start_size in [3, 4, 5]:
            if time.time() > deadline:
                break
            for start_combo in combinations(deg_sorted[:12], start_size):
                if time.time() > deadline:
                    break
                current = list(start_combo)
                # Try to expand
                for candidate in deg_sorted:
                    if candidate in current:
                        continue
                    if time.time() > deadline:
                        break
                    new_current = current + [candidate]
                    if len(new_current) <= best_count:
                        continue
                    paths = greedy_paths(new_current)
                    if paths is not None:
                        current = new_current
                if len(current) > best_count:
                    paths = backtrack_paths(current, time_budget=2.0)
                    if paths is not None:
                        best_count = len(current)
                        result_paths = []
                        for u, v in combinations(current, 2):
                            result_paths.append([int(x) for x in paths[(u, v)]])
                        best_result = {'branches': [int(x) for x in sorted(current)], 'paths': result_paths}
                        break

    if best_result is not None:
        return best_result

    # Fallback: return an edge
    for u, v in P['edges']:
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    return {'branches': [0, 1], 'paths': [[0, 1]]}
