"""Rewritten program: polarity-graph core + optimized extra edge + efficient local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    n = 31
    target_edges = 91

    def build_polarity_graph():
        """Construct a polarity graph on PG(2,5) - C4-free with 90 edges."""
        q = 5
        points = []
        seen = set()
        for x0 in range(q):
            for x1 in range(q):
                for x2 in range(q):
                    if x0 == 0 and x1 == 0 and x2 == 0:
                        continue
                    if x0 != 0:
                        inv = pow(x0, q - 2, q)
                        canon = (1, x1 * inv % q, x2 * inv % q)
                    elif x1 != 0:
                        inv = pow(x1, q - 2, q)
                        canon = (0, 1, x2 * inv % q)
                    else:
                        canon = (0, 0, 1)
                    if canon not in seen:
                        seen.add(canon)
                        points.append(canon)
        assert len(points) == 31

        adj = [set() for _ in range(n)]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (points[i][0] * points[j][2] + points[i][1] * points[j][1] + points[i][2] * points[j][0]) % q == 0:
                    adj[i].add(j)
                    adj[j].add(i)
                    edges.append([i, j])
        return edges, adj

    def count_c4(adj_sets):
        """Count 4-cycles exactly."""
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = len(adj_sets[u] & adj_sets[v])
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 2

    # Step 1: Build the polarity graph (C4-free, 90 edges)
    core_edges, core_adj = build_polarity_graph()

    # Step 2: Add the 91st edge minimizing new C4s
    edge_set = set()
    for e in core_edges:
        edge_set.add(tuple(sorted(e)))

    best_add_c4 = float('inf')
    best_add_edge = None
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edge_set:
                common = len(core_adj[i] & core_adj[j])
                new_c4s = common * (common - 1) // 2
                if new_c4s < best_add_c4:
                    best_add_c4 = new_c4s
                    best_add_edge = (i, j)

    if best_add_edge is not None:
        core_edges.append(list(best_add_edge))
        core_adj[best_add_edge[0]].add(best_add_edge[1])
        core_adj[best_add_edge[1]].add(best_add_edge[0])
        edge_set.add(best_add_edge)

    # Step 3: Local search with edge swaps
    if time.time() < deadline - 3:
        adj = [set() for _ in range(n)]
        for e in core_edges:
            adj[e[0]].add(e[1])
            adj[e[1]].add(e[0])

        edge_set = set(tuple(sorted(e)) for e in core_edges)

        def c4_delta_add(adj_sets, a, b):
            common = len(adj_sets[a] & adj_sets[b])
            return common * (common - 1) // 2

        def c4_delta_remove(adj_sets, a, b):
            common = len(adj_sets[a] & adj_sets[b])
            return common * (common - 1) // 2

        current_c4 = count_c4(adj)

        # Precompute common neighbor counts for all pairs
        def compute_common_matrix(adj_sets):
            common = np.zeros((n, n), dtype=np.int32)
            for u in range(n):
                for v in adj_sets[u]:
                    common[u, v] = len(adj_sets[u] & adj_sets[v])
                    common[v, u] = common[u, v]
            return common

        common_mat = compute_common_matrix(adj)

        def update_common_for_vertex(adj_sets, common, v):
            for x in adj_sets[v]:
                common[v, x] = len(adj_sets[v] & adj_sets[x])
                common[x, v] = common[v, x]

        improved = True
        max_iterations = 30
        iteration = 0

        while improved and time.time() < deadline - 3 and iteration < max_iterations:
            improved = False
            iteration += 1

            edge_list = list(edge_set)
            random.shuffle(edge_list)

            for (u, v) in edge_list:
                if time.time() > deadline - 3:
                    break

                # Delta from removing (u,v)
                common_uv = common_mat[u, v]
                delta_remove = -common_uv * (common_uv - 1) // 2

                # Temporarily remove edge (u,v)
                adj[u].discard(v)
                adj[v].discard(u)
                edge_set.discard((u, v))
                common_mat[u, v] = 0
                common_mat[v, u] = 0

                update_common_for_vertex(adj, common_mat, u)
                update_common_for_vertex(adj, common_mat, v)

                best_delta = 0
                best_new_edge = None

                # Try adding each non-edge
                for a in range(n):
                    for b in range(a + 1, n):
                        if (a, b) == (u, v):
                            continue
                        if (a, b) in edge_set:
                            continue

                        common_ab = common_mat[a, b]
                        delta_add = common_ab * (common_ab - 1) // 2
                        total_delta = delta_remove + delta_add

                        if total_delta < best_delta:
                            best_delta = total_delta
                            best_new_edge = (a, b)

                # Restore (u,v)
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
                common_mat[u, v] = len(adj[u] & adj[v])
                common_mat[v, u] = common_mat[u, v]

                update_common_for_vertex(adj, common_mat, u)
                update_common_for_vertex(adj, common_mat, v)

                if best_new_edge is not None and best_delta < 0:
                    # Perform the swap
                    a, b = best_new_edge
                    adj[u].discard(v)
                    adj[v].discard(u)
                    edge_set.discard((u, v))
                    adj[a].add(b)
                    adj[b].add(a)
                    edge_set.add((a, b))

                    common_mat[u, v] = 0
                    common_mat[v, u] = 0
                    common_mat[a, b] = len(adj[a] & adj[b])
                    common_mat[b, a] = common_mat[a, b]

                    update_common_for_vertex(adj, common_mat, u)
                    update_common_for_vertex(adj, common_mat, v)
                    update_common_for_vertex(adj, common_mat, a)
                    update_common_for_vertex(adj, common_mat, b)

                    current_c4 += best_delta
                    improved = True
                    break

        # Final verification
        current_c4 = count_c4(adj)
        core_edges = [list(sorted(e)) for e in edge_set]

    # Ensure exactly 91 distinct edges
    edge_set = set()
    unique_edges = []
    for e in core_edges:
        key = tuple(sorted(e))
        if key not in edge_set:
            edge_set.add(key)
            unique_edges.append(list(e))

    if len(unique_edges) > target_edges:
        unique_edges = unique_edges[:target_edges]

    if len(unique_edges) < target_edges:
        adj_temp = [set() for _ in range(n)]
        for e in unique_edges:
            adj_temp[e[0]].add(e[1])
            adj_temp[e[1]].add(e[0])
        for i in range(n):
            for j in range(i + 1, n):
                if len(unique_edges) >= target_edges:
                    break
                if (i, j) not in edge_set:
                    unique_edges.append([i, j])
                    edge_set.add((i, j))
            if len(unique_edges) >= target_edges:
                break

    return unique_edges[:target_edges]
# EVOLVE-BLOCK-END
