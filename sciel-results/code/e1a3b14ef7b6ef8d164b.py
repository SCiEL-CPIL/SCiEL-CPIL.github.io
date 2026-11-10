import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    def try_branch_set_flow(branches):
        """Use max flow with node-splitting to find vertex-disjoint paths for all pairs."""
        k = len(branches)
        if k < 2:
            return None
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        num_paths = len(pairs)

        S = 2 * n
        T = 2 * n + 1
        total_nodes = 2 * n + 2

        cap = [[0] * total_nodes for _ in range(total_nodes)]

        # Vertex capacity edges: vin -> vout
        for v in range(n):
            vin = 2 * v
            vout = 2 * v + 1
            if v in branch_set:
                cap[vin][vout] = k
            else:
                cap[vin][vout] = 1

        # Graph edges (bidirectional)
        for u, v in P['edges']:
            uout = 2 * u + 1
            vin = 2 * v
            vout = 2 * v + 1
            uin = 2 * u
            cap[uout][vin] = 1
            cap[vout][uin] = 1

        # Source/sink connections
        for a, b in pairs:
            a_in = 2 * a
            b_out = 2 * b + 1
            cap[S][a_in] += 1
            cap[b_out][T] += 1

        # Max flow with path tracking
        flow = 0
        all_paths = []

        while True:
            parent = [-1] * total_nodes
            parent[S] = -2
            queue = deque([S])
            while queue and parent[T] == -1:
                u = queue.popleft()
                for v in range(total_nodes):
                    if parent[v] == -1 and cap[u][v] > 0:
                        parent[v] = u
                        queue.append(v)
            if parent[T] == -1:
                break

            # Trace path
            path_nodes = []
            v = T
            while v != S:
                path_nodes.append(v)
                v = parent[v]
            path_nodes.append(S)
            path_nodes.reverse()

            # Update residual
            v = T
            while v != S:
                u = parent[v]
                cap[u][v] -= 1
                cap[v][u] += 1
                v = u
            flow += 1

            internal = path_nodes[1:-1]
            all_paths.append(internal)

        if flow < num_paths:
            return None

        # Convert split-graph paths to original vertex paths
        result_paths = []
        for sp in all_paths:
            verts = []
            for node in sp:
                v = node // 2
                if not verts or verts[-1] != v:
                    verts.append(v)
            result_paths.append(verts)

        return result_paths

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

    best_branches = None
    best_paths = None

    initial_clique = find_max_clique_greedy(adj, n)
    if len(initial_clique) >= 2:
        paths_result = try_branch_set_flow(initial_clique)
        if paths_result is not None:
            best_branches = initial_clique[:]
            best_paths = paths_result

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
                paths_result = try_branch_set_flow(new_branches)
                if paths_result is not None:
                    current_branches = new_branches
                    best_branches = current_branches[:]
                    best_paths = paths_result
                    expanded = True
                    break

    import random
    rng = random.Random(123)
    for size in range(n, 1, -1):
        if time.time() > deadline:
            break
        if best_branches is not None and size <= len(best_branches):
            break
        for attempt in range(200):
            if time.time() > deadline:
                break
            random_set = rng.sample(range(n), size)
            paths_result = try_branch_set_flow(random_set)
            if paths_result is not None:
                best_branches = random_set[:]
                best_paths = paths_result
                break

    if best_branches is None or len(best_branches) < 2:
        u, v = P['edges'][0]
        best_branches = [int(u), int(v)]
        best_paths = [[int(u), int(v)]]
    else:
        best_branches = [int(v) for v in best_branches]
        best_paths = [[int(v) for v in path] for path in best_paths]

    return {'branches': best_branches, 'paths': best_paths}
