"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import collections
import random

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    n = P['n']
    edges = P['edges']
    adj = collections.defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj = {i: sorted(adj[i]) for i in range(n)}

    # Precompute all-pairs shortest path distances
    dist = [[n + 1 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        queue = collections.deque([i])
        while queue:
            node = queue.popleft()
            for nei in adj[node]:
                if dist[i][nei] == n + 1:
                    dist[i][nei] = dist[i][node] + 1
                    queue.append(nei)

    # ---- Max-flow with vertex splitting (Dinic's algorithm) ----
    # Vertex splitting: each vertex v -> v_in (2v), v_out (2v+1)
    # Edge v_in -> v_out has capacity 1 for non-branch vertices, INF for branch vertices
    # Original edge (u,v) becomes u_out -> v_in and v_out -> u_in with capacity 1

    def dinic_maxflow(num_nodes, edges_list, source, sink):
        """
        edges_list: list of (u, v, capacity)
        Returns max flow value.
        """
        INF = float('inf')
        # Build residual graph
        graph = collections.defaultdict(list)
        for u, v, cap in edges_list:
            if cap > 0:
                graph[u].append([v, cap, len(graph[v])])
                graph[v].append([u, 0, len(graph[u]) - 1])

        flow = 0
        while True:
            # BFS to find augmenting paths
            level = [-1] * num_nodes
            level[source] = 0
            queue = collections.deque([source])
            while queue:
                u = queue.popleft()
                for v, cap, _ in graph[u]:
                    if cap > 0 and level[v] < 0:
                        level[v] = level[u] + 1
                        queue.append(v)
            if level[sink] < 0:
                break
            # DFS to send flow
            iter_arr = [0] * num_nodes
            def dfs(u, f):
                if u == sink:
                    return f
                for i in range(iter_arr[u], len(graph[u])):
                    iter_arr[u] = i
                    v, cap, rev = graph[u][i]
                    if cap > 0 and level[v] == level[u] + 1:
                        d = dfs(v, min(f, cap))
                        if d > 0:
                            graph[u][i][1] -= d
                            graph[v][rev][1] += d
                            return d
                return 0
            while True:
                f = dfs(source, INF)
                if f == 0:
                    break
                flow += f
        return flow

    def build_flow_network(branches):
        """
        Build flow network for testing if branches admit internally disjoint paths.
        Returns (num_nodes, edges_list, source, sink, pair_info)
        where pair_info maps flow edge index to (branch_i, branch_j)
        """
        branch_set = set(branches)
        k = len(branches)
        # Node numbering:
        # v_in = 2*v, v_out = 2*v+1 for v in range(n)
        # source = 2*n, sink = 2*n+1
        # Additional: for each pair (i,j), we need a dedicated edge
        # Actually, we can use the standard construction:
        # - For each vertex v: v_in -> v_out (cap 1 if not branch, INF if branch)
        # - For each edge (u,v): u_out -> v_in (cap 1) and v_out -> u_in (cap 1)
        # - For each pair (i,j) of branches: source -> branch_i_in (cap 1), branch_j_out -> sink (cap 1)
        # Wait, this doesn't quite work for K_k subdivision.
        #
        # Better approach: For each pair (i,j), create a separate source-sink pair.
        # Actually, the standard way: 
        # - Split vertices: v_in, v_out
        # - v_in -> v_out: cap 1 for non-branch, cap INF for branch
        # - For each original edge (u,v): u_out -> v_in (cap 1), v_out -> u_in (cap 1)
        # - For each branch b: source -> b_in (cap deg_in), b_out -> sink (cap deg_out)
        #   where deg_in = deg_out = k-1 (each branch connects to k-1 others)
        #
        # This gives max flow = k(k-1)/2 iff the subdivision exists.

        num_nodes = 2 * n + 2
        source = 2 * n
        sink = 2 * n + 1
        INF = k * (k - 1) + 1  # Large enough

        edges_list = []
        # Vertex splitting edges
        for v in range(n):
            cap = INF if v in branch_set else 1
            edges_list.append((2 * v, 2 * v + 1, cap))

        # Original edges
        for u, v in edges:
            edges_list.append((2 * u + 1, 2 * v, 1))
            edges_list.append((2 * v + 1, 2 * u, 1))

        # Source/sink connections for branches
        for b in branches:
            edges_list.append((source, 2 * b, k - 1))
            edges_list.append((2 * b + 1, sink, k - 1))

        return num_nodes, edges_list, source, sink

    def verify_and_extract(branches):
        """
        Verify if branches form a K_k subdivision using max-flow.
        If yes, extract the paths.
        Returns (success, paths_dict) where paths_dict maps (i,j) to path.
        """
        k = len(branches)
        if k < 2:
            return False, {}

        num_nodes, edges_list, source, sink = build_flow_network(branches)
        maxflow = dinic_maxflow(num_nodes, edges_list, source, sink)
        required = k * (k - 1) // 2

        if maxflow < required:
            return False, {}

        # Extract paths from the flow
        # We need to trace which flow went from which branch to which branch
        # This is tricky with the shared network. Let's use a different approach:
        # For each pair, run a separate max-flow with unit capacity.
        # Actually, let's just use the backtracking approach for extraction,
        # but use the flow test as a filter.

        # For extraction, use BFS-based path finding with backtracking
        branch_set = set(branches)
        branch_to_idx = {b: i for i, b in enumerate(branches)}
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        pair_order = sorted(pairs, key=lambda p: dist[branches[p[0]]][branches[p[1]]])

        def get_valid_path(u, v, used_internal):
            visited = set()
            visited.add(u)
            parent = {u: None}
            queue = collections.deque([u])
            while queue:
                curr = queue.popleft()
                if curr == v:
                    path = []
                    node = v
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                for nei in adj[curr]:
                    if nei in visited:
                        continue
                    if nei in branch_set and nei != v:
                        continue
                    if nei in used_internal:
                        continue
                    visited.add(nei)
                    parent[nei] = curr
                    queue.append(nei)
            return None

        assigned_paths = {}
        used_internal = set()

        def backtrack(idx):
            if idx == len(pair_order):
                return True
            i, j = pair_order[idx]
            u = branches[i]
            v = branches[j]
            path = get_valid_path(u, v, used_internal)
            if path is None:
                return False
            new_used = set(used_internal)
            for node in path[1:-1]:
                new_used.add(node)
            assigned_paths[(i, j)] = path
            if backtrack(idx + 1):
                return True
            return False

        if backtrack(0):
            return True, assigned_paths
        return False, {}

    # ---- Find max clique ----
    def find_max_clique():
        max_clique = []
        candidates = list(range(n))

        def expand(clique, candidates):
            nonlocal max_clique
            if len(clique) > len(max_clique):
                max_clique = clique[:]
            if len(candidates) == 0:
                return
            v = candidates[0]
            new_candidates = [u for u in candidates[1:] if u in adj[v]]
            expand(clique + [v], new_candidates)
            expand(clique, candidates[1:])

        expand([], candidates)
        return max_clique

    max_clique = find_max_clique()
    best_k = len(max_clique)
    best_result = {'branches': list(max_clique),
                   'paths': [[max_clique[i], max_clique[j]] for i in range(best_k) for j in range(i + 1, best_k)]}

    # ---- Greedy growth: try to add vertices to the clique ----
    deg = [len(adj[i]) for i in range(n)]
    sorted_by_deg = sorted(range(n), key=lambda x: (-deg[x], x))

    # Try to grow from the max clique
    current_branches = list(max_clique)
    for _ in range(20):  # Try up to 20 additions
        if time.time() > deadline:
            break
        best_add = None
        best_add_score = 0
        for v in sorted_by_deg:
            if v in current_branches:
                continue
            if time.time() > deadline:
                break
            new_branches = current_branches + [v]
            k = len(new_branches)
            # Quick check: does the flow test pass?
            num_nodes, edges_list, source, sink = build_flow_network(new_branches)
            maxflow = dinic_maxflow(num_nodes, edges_list, source, sink)
            required = k * (k - 1) // 2
            if maxflow >= required:
                # Verify with extraction
                success, paths_dict = verify_and_extract(new_branches)
                if success:
                    best_add = v
                    break
        if best_add is not None:
            current_branches.append(best_add)
            best_k = len(current_branches)
            success, paths_dict = verify_and_extract(current_branches)
            if success:
                result_paths = []
                for i in range(best_k):
                    for j in range(i + 1, best_k):
                        result_paths.append(paths_dict[(i, j)])
                best_result = {'branches': list(current_branches), 'paths': result_paths}
        else:
            break

    # ---- Random multi-start search ----
    random.seed(42)
    for k in range(best_k + 1, min(n, 12) + 1):
        if time.time() > deadline:
            break
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            # Sample from top-degree vertices
            pool_size = min(n, k * 2 + 2)
            pool = sorted_by_deg[:pool_size]
            sample = random.sample(pool, k)
            sample.sort()
            # Quick flow test
            num_nodes, edges_list, source, sink = build_flow_network(sample)
            maxflow = dinic_maxflow(num_nodes, edges_list, source, sink)
            required = k * (k - 1) // 2
            if maxflow >= required:
                success, paths_dict = verify_and_extract(sample)
                if success:
                    result_paths = []
                    for i in range(k):
                        for j in range(i + 1, k):
                            result_paths.append(paths_dict[(i, j)])
                    best_result = {'branches': list(sample), 'paths': result_paths}
                    best_k = k
                    found = True
                    break
        if found:
            continue
        else:
            break

    # Ensure types are correct
    if best_result:
        best_result['branches'] = [int(b) for b in best_result['branches']]
        new_paths = []
        for p in best_result['paths']:
            new_paths.append([int(x) for x in p])
        best_result['paths'] = new_paths

    return best_result

# EVOLVE-BLOCK-END
