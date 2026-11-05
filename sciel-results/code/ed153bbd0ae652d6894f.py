import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin before 90s kill

    n = 24
    edge_order = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

    # Map edge to position
    pos = {}
    for i, (u, v) in enumerate(edge_order):
        pos[(u, v)] = i + 1
        pos[(v, u)] = i + 1

    def check_hamilton_cycle(K, rng, max_time=0.4):
        """Check if G_K has a Hamilton cycle using DFS with pruning."""
        check_start = time.time()

        # Build adjacency lists
        adj = [set() for _ in range(n)]
        for i in range(K):
            u, v = edge_order[i]
            adj[u].add(v)
            adj[v].add(u)

        # Quick checks
        for v in range(n):
            if len(adj[v]) < 2:
                return None

        # Connectivity check via BFS
        visited = [False] * n
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            u = queue.pop()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    queue.append(v)
        if count < n:
            return None

        # DFS with Warnsdorff's rule and pruning
        visited_arr = [False] * n
        path = []

        def dfs(u, depth):
            if time.time() - check_start > max_time:
                return False

            if depth == n:
                return start in adj[u]

            visited_arr[u] = True
            path.append(u)

            # Get unvisited neighbors
            unvisited_neighbors = [v for v in adj[u] if not visited_arr[v]]

            if not unvisited_neighbors:
                visited_arr[u] = False
                path.pop()
                return False

            # Pruning: check if any unvisited vertex (not in path) has only one
            # unvisited neighbor and that neighbor is not u (dead end)
            # Actually, simpler: check if any unvisited vertex has degree 1 in
            # the remaining graph (only connected to vertices in path)
            # This is expensive, so do it every few steps

            if depth % 4 == 0 and depth > 2:
                for v in range(n):
                    if not visited_arr[v]:
                        unvisited_neighbors_v = [w for w in adj[v] if not visited_arr[w]]
                        if len(unvisited_neighbors_v) == 1:
                            # This vertex can only be reached from one direction
                            # It must be next or it becomes unreachable
                            # Actually this is a soft constraint, skip for now
                            pass

            # Sort by Warnsdorff: fewer unvisited neighbors first
            def unvisited_deg(v):
                return sum(1 for w in adj[v] if not visited_arr[w])

            unvisited_neighbors.sort(key=unvisited_deg)

            # Shuffle ties for randomization
            i = 0
            while i < len(unvisited_neighbors):
                j = i
                while j < len(unvisited_neighbors) and unvisited_deg(unvisited_neighbors[j]) == unvisited_deg(unvisited_neighbors[i]):
                    j += 1
                group = unvisited_neighbors[i:j]
                rng.shuffle(group)
                unvisited_neighbors[i:j] = group
                i = j

            for v in unvisited_neighbors:
                if not visited_arr[v]:
                    if dfs(v, depth + 1):
                        return True

            visited_arr[u] = False
            path.pop()
            return False

        # Try multiple starting vertices
        for start in range(n):
            if time.time() - check_start > max_time:
                break

            visited_arr = [False] * n
            path = []

            # Redefine dfs with correct start
            def dfs2(u, depth):
                if time.time() - check_start > max_time:
                    return False

                if depth == n:
                    return start in adj[u]

                visited_arr[u] = True
                path.append(u)

                unvisited_neighbors = [v for v in adj[u] if not visited_arr[v]]

                if not unvisited_neighbors:
                    visited_arr[u] = False
                    path.pop()
                    return False

                def unvisited_deg2(v):
                    return sum(1 for w in adj[v] if not visited_arr[w])

                unvisited_neighbors.sort(key=unvisited_deg2)

                i = 0
                while i < len(unvisited_neighbors):
                    j = i
                    while j < len(unvisited_neighbors) and unvisited_deg2(unvisited_neighbors[j]) == unvisited_deg2(unvisited_neighbors[i]):
                        j += 1
                    group = unvisited_neighbors[i:j]
                    rng.shuffle(group)
                    unvisited_neighbors[i:j] = group
                    i = j

                for v in unvisited_neighbors:
                    if not visited_arr[v]:
                        if dfs2(v, depth + 1):
                            return True

                visited_arr[u] = False
                path.pop()
                return False

            if dfs2(start, 0):
                # Rotate so it starts at 0
                idx0 = path.index(0)
                rotated = path[idx0:] + path[:idx0]
                return rotated

        return None

    # Binary search for minimum K
    # Lower bound: n (minimum edges for a cycle)
    # Upper bound: len(edge_order) (all edges)
    lo = n
    hi = len(edge_order)

    # First, verify hi works
    result_hi = None
    for seed in range(5):
        rng = random.Random(seed * 10000 + 999)
        result_hi = check_hamilton_cycle(hi, rng, max_time=1.0)
        if result_hi is not None:
            break

    if result_hi is None:
        return list(range(n))

    # Binary search
    best_result = result_hi
    best_K = hi

    while lo < hi:
        if time.time() > deadline:
            break

        mid = (lo + hi) // 2

        found = None
        for seed in range(15):
            if time.time() > deadline:
                break
            rng = random.Random(seed * 10000 + mid * 7)
            result = check_hamilton_cycle(mid, rng, max_time=0.3)
            if result is not None:
                found = result
                break

        if found is not None:
            best_result = found
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1

    if best_result is not None:
        return best_result

    return list(range(n))

print(construct())
