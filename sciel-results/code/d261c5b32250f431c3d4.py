import time
import math
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin for safety

    N = 256

    # Squares up to 2*256 = 512
    squares = set()
    i = 1
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1

    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)

    node_index = {v: i for i, v in enumerate(valid_nodes)}
    node_set = set(valid_nodes)
    nv = len(valid_nodes)

    # Build conflict graph using bitsets for speed
    conflicts = [0] * nv
    for i_idx in range(nv):
        v = valid_nodes[i_idx]
        for i_j in range(i_idx + 1, nv):
            u = valid_nodes[i_j]
            if v + u in squares:
                conflicts[i_idx] |= (1 << i_j)
                conflicts[i_j] |= (1 << i_idx)

    # Check if graph is bipartite
    def is_bipartite_and_get_partition():
        color = [-1] * nv
        partition = [0] * nv
        for i in range(nv):
            if color[i] == -1:
                color[i] = 0
                partition[i] = 0
                q = deque([i])
                while q:
                    u = q.popleft()
                    mask = conflicts[u]
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        mask ^= lsb
                        if color[v] == -1:
                            color[v] = 1 - color[u]
                            partition[v] = color[v]
                            q.append(v)
                        elif color[v] == color[u]:
                            return None, None
        return partition, color

    partition, _ = is_bipartite_and_get_partition()

    if partition is not None:
        # Graph is bipartite: use max-flow to find exact MIS
        # MIS in bipartite = |V| - min_vertex_cover = |V| - max_matching
        # Using max flow: source->U (cap 1), U->V (cap INF), V->sink (cap 1)
        source = nv
        sink = nv + 1
        n_nodes = nv + 2

        graph = [[] for _ in range(n_nodes)]

        def add_edge(u, v, cap):
            graph[u].append([v, cap, len(graph[v])])
            graph[v].append([u, 0, len(graph[u]) - 1])

        for i in range(nv):
            if partition[i] == 0:
                add_edge(source, i, 1)
            else:
                add_edge(i, sink, 1)

        INF_CAP = 10**9
        for i in range(nv):
            if partition[i] == 0:
                mask = conflicts[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    mask ^= lsb
                    if partition[j] == 1:
                        add_edge(i, j, INF_CAP)

        # Dinic's algorithm
        level = [0] * n_nodes
        it = [0] * n_nodes

        def bfs_flow():
            for i in range(n_nodes):
                level[i] = -1
            level[source] = 0
            q = deque([source])
            while q:
                u = q.popleft()
                for v, cap, rev in graph[u]:
                    if cap > 0 and level[v] < 0:
                        level[v] = level[u] + 1
                        q.append(v)
            return level[sink] >= 0

        def dfs_flow(u, f):
            if u == sink:
                return f
            for i in range(it[u], len(graph[u])):
                it[u] = i
                v, cap, rev = graph[u][i]
                if cap > 0 and level[u] < level[v]:
                    d = dfs_flow(v, min(f, cap))
                    if d > 0:
                        graph[u][i][1] -= d
                        graph[v][rev][1] += d
                        return d
            return 0

        while bfs_flow():
            for i in range(n_nodes):
                it[i] = 0
            while True:
                f = dfs_flow(source, 10**9)
                if f == 0:
                    break

        # Find reachable nodes from Source in residual graph
        reachable = [False] * n_nodes
        q = deque([source])
        reachable[source] = True
        while q:
            u = q.popleft()
            for v, cap, rev in graph[u]:
                if cap > 0 and not reachable[v]:
                    reachable[v] = True
                    q.append(v)

        # MIS = (U ∩ Reachable) ∪ (V ∩ Unreachable)
        result_nodes = []
        for i in range(nv):
            if partition[i] == 0:
                if reachable[i]:
                    result_nodes.append(valid_nodes[i])
            else:
                if not reachable[i]:
                    result_nodes.append(valid_nodes[i])

        # Verify
        res_set = set(result_nodes)
        for a in res_set:
            if 2 * a in squares:
                return [valid_nodes[0]]
            for b in res_set:
                if a <= b and (a + b) in squares:
                    return [valid_nodes[0]]

        if result_nodes:
            return result_nodes

    # If not bipartite or flow failed, use heuristic approach
    def popcount(x):
        return bin(x).count('1')

    def greedy_mis_bitset(order_indices):
        chosen_mask = 0
        for idx in order_indices:
            if (conflicts[idx] & chosen_mask) == 0:
                chosen_mask |= (1 << idx)
        return chosen_mask

    def local_search_bitset(mask, max_iters=3000):
        mask = mask
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break

            # Try adding each non-chosen node
            for i in range(nv):
                if mask & (1 << i):
                    continue
                if (conflicts[i] & mask) == 0:
                    mask |= (1 << i)
                    improved = True

            # Try swaps
            for i in range(nv):
                if mask & (1 << i):
                    continue
                conf_in_mask = conflicts[i] & mask
                if conf_in_mask == 0:
                    mask |= (1 << i)
                    improved = True
                    continue
                temp = conf_in_mask
                while temp:
                    r_bit = temp & (-temp)
                    r_idx = r_bit.bit_length() - 1
                    if (conflicts[i] & (mask ^ r_bit)) == 0:
                        mask = (mask & ~r_bit) | (1 << i)
                        improved = True
                        break
                    temp ^= r_bit

        return mask

    def mask_to_set(mask):
        s = set()
        for i in range(nv):
            if mask & (1 << i):
                s.add(valid_nodes[i])
        return s

    best_mask = 0
    best_count = 0

    random.seed(42)

    # Multiple heuristic orderings
    order_by_degree = sorted(range(nv), key=lambda i: popcount(conflicts[i]))
    mask = greedy_mis_bitset(order_by_degree)
    mask = local_search_bitset(mask, max_iters=2000)
    c = popcount(mask)
    if c > best_count:
        best_count = c
        best_mask = mask

    order_by_degree_desc = sorted(range(nv), key=lambda i: popcount(conflicts[i]), reverse=True)
    mask = greedy_mis_bitset(order_by_degree_desc)
    mask = local_search_bitset(mask, max_iters=2000)
    c = popcount(mask)
    if c > best_count:
        best_count = c
        best_mask = mask

    mask = greedy_mis_bitset(list(range(nv)))
    mask = local_search_bitset(mask, max_iters=2000)
    c = popcount(mask)
    if c > best_count:
        best_count = c
        best_mask = mask

    mask = greedy_mis_bitset(list(range(nv - 1, -1, -1)))
    mask = local_search_bitset(mask, max_iters=2000)
    c = popcount(mask)
    if c > best_count:
        best_count = c
        best_mask = mask

    # Random orderings
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        order = list(range(nv))
        random.shuffle(order)
        mask = greedy_mis_bitset(order)
        mask = local_search_bitset(mask, max_iters=1500)
        c = popcount(mask)
        if c > best_count:
            best_count = c
            best_mask = mask

    # Final polish
    if time.time() < deadline - 3:
        mask = best_mask
        for _ in range(30):
            if time.time() > deadline:
                break
            mask = local_search_bitset(mask, max_iters=1000)
            c = popcount(mask)
            if c > best_count:
                best_count = c
                best_mask = mask
            else:
                # Perturb
                chosen_indices = [i for i in range(nv) if best_mask & (1 << i)]
                if chosen_indices:
                    random.shuffle(chosen_indices)
                    perturbed = best_mask
                    for idx in chosen_indices[-2:]:
                        perturbed &= ~(1 << idx)
                    for i in range(nv):
                        if (conflicts[i] & perturbed) == 0:
                            perturbed |= (1 << i)
                    c = popcount(perturbed)
                    if c > best_count:
                        best_count = c
                        best_mask = perturbed

    result = mask_to_set(best_mask)
    if not result:
        result = {valid_nodes[0]}

    return list(result)
