"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import gcd

    deadline = time.time() + 55.0

    nums = list(range(2, 257))
    n = len(nums)

    # Precompute conflicts.
    # conflict[i] is a set of indices j such that lcm(nums[i], nums[j]) <= 256.
    conflict = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = gcd(nums[i], nums[j])
            if (nums[i] // g) * nums[j] <= 256:
                conflict[i].add(j)
                conflict[j].add(i)

    # Determine if the conflict graph is bipartite.
    # A graph is bipartite if it contains no odd-length cycles.
    # We use BFS to check for bipartiteness.
    color = [-1] * n
    is_bipartite = True
    for i in range(n):
        if color[i] == -1:
            color[i] = 0
            queue = [i]
            while queue:
                u = queue.pop(0)
                for v in conflict[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        is_bipartite = False
                        break
                if not is_bipartite:
                    break
        if not is_bipartite:
            break

    if is_bipartite:
        # Solve Maximum Weight Independent Set on a bipartite graph.
        # This is equivalent to Minimum Weight Vertex Cover, which can be solved via Max Flow / Min Cut.
        # Let U be the set of nodes with color 0, and V be the set with color 1.
        # We construct a flow network:
        # Source -> u (for u in U) with capacity w(u) = 1/nums[u]
        # v -> Sink (for v in V) with capacity w(v) = 1/nums[v]
        # u -> v (for each edge (u,v) in conflict) with capacity infinity
        #
        # The minimum vertex cover is:
        # {u in U : u is reachable from Source in residual graph} U {v in V : v is NOT reachable from Source}
        # The maximum independent set is the complement of the minimum vertex cover.

        from collections import defaultdict, deque

        # Build adjacency list for flow network
        # Nodes: 0 = Source, 1..n = original nodes, n+1 = Sink
        # To simplify indexing, let's map original indices to flow node indices.
        # Source = 0
        # Node i = i + 1
        # Sink = n + 1

        source = 0
        sink = n + 1
        num_nodes = n + 2

        # Edge list for Dinic's algorithm
        # Each edge is represented as [to, cap, rev_index]
        # We need to add reverse edges for residual graph.
        # Let's use a list of lists: graph[u] = [(v, cap, rev_index_in_graph_v), ...]

        graph = [[] for _ in range(num_nodes)]

        def add_edge(u, v, cap):
            graph[u].append([v, cap, len(graph[v])])
            graph[v].append([u, 0, len(graph[u]) - 1])

        # Add edges from source to U (color 0)
        for i in range(n):
            if color[i] == 0:
                add_edge(source, i + 1, 1.0 / nums[i])

        # Add edges from V (color 1) to sink
        for i in range(n):
            if color[i] == 1:
                add_edge(i + 1, sink, 1.0 / nums[i])

        # Add edges between U and V for each conflict
        for i in range(n):
            if color[i] == 0:
                for j in conflict[i]:
                    if color[j] == 1:
                        add_edge(i + 1, j + 1, float('inf'))

        # Dinic's Algorithm
        def dinic():
            flow = 0
            while True:
                # BFS to find levels
                level = [-1] * num_nodes
                level[source] = 0
                q = deque([source])
                while q:
                    u = q.popleft()
                    for v, cap, rev in graph[u]:
                        if cap > 0 and level[v] < 0:
                            level[v] = level[u] + 1
                            q.append(v)
                if level[sink] < 0:
                    break

                # DFS to find blocking flow
                it = [0] * num_nodes

                def dfs(u, f):
                    if u == sink:
                        return f
                    for i in range(it[u], len(graph[u])):
                        it[u] = i
                        v, cap, rev = graph[u][i]
                        if cap > 0 and level[v] == level[u] + 1:
                            d = dfs(v, min(f, cap))
                            if d > 0:
                                graph[u][i][1] -= d
                                graph[v][rev][1] += d
                                return d
                    return 0

                while True:
                    f = dfs(source, float('inf'))
                    if f == 0:
                        break
                    flow += f
            return flow

        dinic()

        # Find reachable nodes in residual graph from source
        reachable = [False] * num_nodes
        reachable[source] = True
        q = deque([source])
        while q:
            u = q.popleft()
            for v, cap, rev in graph[u]:
                if cap > 0 and not reachable[v]:
                    reachable[v] = True
                    q.append(v)

        # Minimum Vertex Cover
        vertex_cover = set()
        for i in range(n):
            node_idx = i + 1
            if color[i] == 0 and reachable[node_idx]:
                vertex_cover.add(i)
            elif color[i] == 1 and not reachable[node_idx]:
                vertex_cover.add(i)

        # Maximum Independent Set
        independent_set = set(range(n)) - vertex_cover

        # Verify validity
        valid = True
        for i in independent_set:
            for j in conflict[i]:
                if j in independent_set and i < j:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            return [nums[i] for i in independent_set]
        else:
            # Fallback if verification fails (should not happen)
            return [nums[i] for i in independent_set]

    else:
        # Fallback to heuristic if graph is not bipartite
        def is_valid_add(selected, i):
            if len(selected) < len(conflict[i]):
                return not selected.intersection(conflict[i])
            else:
                return not conflict[i].intersection(selected)

        order = sorted(range(n), key=lambda i: nums[i])
        initial_set = set()
        for i in order:
            if is_valid_add(initial_set, i):
                initial_set.add(i)

        # Local search
        def local_search(init_set, max_iters=20):
            best = set(init_set)
            best_score = sum(1.0 / nums[i] for i in best)

            for _ in range(max_iters):
                if time.time() > deadline:
                    break

                improved = False

                # Try adding one element
                best_add = -1
                best_add_val = 0.0
                for i in range(n):
                    if i in best:
                        continue
                    if is_valid_add(best, i):
                        val = 1.0 / nums[i]
                        if val > best_add_val:
                            best_add_val = val
                            best_add = i

                if best_add != -1:
                    best.add(best_add)
                    best_score += best_add_val
                    improved = True

                # Try swaps
                if time.time() < deadline - 5.0:
                    best_swap = None
                    best_swap_gain = 0.0

                    for i in best:
                        if time.time() > deadline:
                            break
                        for j in range(n):
                            if j in best:
                                continue
                            if nums[j] >= nums[i]:
                                continue

                            inter = conflict[j] & best
                            if i in inter:
                                if len(inter) > 1:
                                    continue
                            else:
                                if len(inter) > 0:
                                    continue

                            gain = 1.0 / nums[j] - 1.0 / nums[i]
                            if gain > best_swap_gain:
                                best_swap_gain = gain
                                best_swap = (i, j)

                    if best_swap is not None:
                        i, j = best_swap
                        best.remove(i)
                        best.add(j)
                        best_score += best_swap_gain
                        improved = True

                if not improved:
                    break

            return best, best_score

        ls_set, ls_score = local_search(initial_set, max_iters=10)
        return [nums[i] for i in ls_set]

# EVOLVE-BLOCK-END
