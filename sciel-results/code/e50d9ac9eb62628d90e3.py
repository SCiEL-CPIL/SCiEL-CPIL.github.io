import time
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin below the 90s kill

    p = 101
    invs = [pow(x, -1, p) for x in range(1, 11)]

    # Precompute all non-empty subsets of {1..10}
    # For each subset, record: (size, sum_mod_p, denoms_list)
    # Group by size for efficient access
    by_size = [[] for _ in range(11)]  # by_size[k] = list of (sum_mod_p, denoms)
    for mask in range(1, 1 << 10):
        size = bin(mask).count('1')
        s = 0
        denoms = []
        for i in range(10):
            if mask & (1 << i):
                s = (s + invs[i]) % p
                denoms.append(i + 1)
        by_size[size].append((s, denoms))

    # Find min_size[r] = minimum subset size that achieves residue r
    min_size = [11] * p
    for size in range(1, 11):
        for s, denoms in by_size[size]:
            if size < min_size[s]:
                min_size[s] = size

    min_possible = max(min_size) if all(min_size[r] < 11 for r in range(p)) else 11

    # Hopcroft-Karp with iterative DFS to avoid recursion depth issues
    def hopcroft_karp(adj, n_left, n_right):
        INF = float('inf')
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left

        def bfs():
            queue = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = INF
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if pair_v[v] == -1:
                        found = True
                    elif dist[pair_v[v]] == INF:
                        dist[pair_v[v]] = dist[u] + 1
                        queue.append(pair_v[v])
            return found

        def dfs(u):
            # Iterative DFS using explicit stack
            stack = [(u, 0)]  # (node, next_adj_index)
            parent = {}  # v -> (u, v) to track path
            while stack:
                node, idx = stack[-1]
                if idx < len(adj[node]):
                    stack[-1] = (node, idx + 1)
                    v = adj[node][idx]
                    if pair_v[v] == -1:
                        # Found augmenting path, apply it
                        # Trace back through parent to find the path
                        # We need to track the full path
                        # Let's use a different approach: recursive but with path tracking
                        # Actually, let's just do it recursively but carefully
                        pair_u[node] = v
                        pair_v[v] = node
                        return True
                    elif dist[pair_v[v]] == dist[node] + 1:
                        next_u = pair_v[v]
                        stack.append((next_u, 0))
                        parent[next_u] = (node, v)
                else:
                    stack.pop()
                    dist[node] = INF
                    if parent.get(node) is not None:
                        del parent[node]
            return False

        # Use recursive DFS but with path reconstruction
        def dfs_recursive(u, visited_v):
            for v in adj[u]:
                if v in visited_v:
                    continue
                visited_v.add(v)
                if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs_recursive(pair_v[v], visited_v)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = INF
            return False

        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    visited_v = set()
                    if dfs_recursive(u, visited_v):
                        matching += 1
        return pair_u, pair_v

    # Try K from min_possible upward
    best_result = None
    for K in range(min_possible, 11):
        if time.time() > deadline:
            break

        # Collect all subsets with size <= K
        right_subsets = []
        for size in range(1, K + 1):
            for s, denoms in by_size[size]:
                right_subsets.append((s, denoms))

        n_right = len(right_subsets)

        # Build adjacency: for each residue r, which right nodes have sum == r
        adj = [[] for _ in range(p)]
        for j, (s, denoms) in enumerate(right_subsets):
            adj[s].append(j)

        # Run Hopcroft-Karp
        pair_u, pair_v = hopcroft_karp(adj, p, n_right)

        # Check if all residues are matched
        if all(pair_u[r] != -1 for r in range(p)):
            # Perfect matching found! Construct the result
            result = [None] * p
            for r in range(p):
                j = pair_u[r]
                result[r] = right_subsets[j][1]
            best_result = result
            break

    # If no perfect matching found, fall back to greedy
    if best_result is None:
        best_result = [None] * p
        for r in range(p):
            found = False
            for size in range(1, 11):
                for s, denoms in by_size[size]:
                    if s == r:
                        best_result[r] = denoms
                        found = True
                        break
                if found:
                    break
            if not found:
                best_result[r] = list(range(1, 11))

    return best_result
