import time
import random
from itertools import combinations
from collections import deque

def construct():
    t_start = time.time()
    DEADLINE = 83.0

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_list = [sorted(adj[i]) for i in range(n)]
    adj_bool = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj_bool[u][v] = True
        adj_bool[v][u] = True

    def time_up():
        return time.time() - t_start > DEADLINE

    # ---------- Max-flow (Dinic) on an explicit edge list ----------
    def maxflow(n_nodes, edge_list, s, t):
        # build adjacency with edge indices
        head = [[] for _ in range(n_nodes)]
        for idx, (a, b) in enumerate(edge_list):
            head[a].append(idx)
            head[b].append(idx ^ 1)
        cap = [1] * (2 * len(edge_list))
        level = [0] * n_nodes
        it = [0] * n_nodes

        def bfs():
            for i in range(n_nodes):
                level[i] = -1
            dq = deque([s])
            level[s] = 0
            while dq:
                u = dq.popleft()
                for ei in head[u]:
                    if cap[ei] > 0:
                        v = (ei ^ 1) if (ei // 2) * 2 == ei else ei
                        # compute neighbor
                        a, b = edge_list[ei // 2]
                        v = b if a == u else a
                        if level[v] < 0:
                            level[v] = level[u] + 1
                            dq.append(v)
            return level[t] >= 0

        def dfs(u, f):
            if u == t:
                return f
            for i in range(it[u], len(head[u])):
                it[u] = i
                ei = head[u][i]
                if cap[ei] <= 0:
                    continue
                a, b = edge_list[ei // 2]
                v = b if a == u else a
                if level[v] != level[u] + 1:
                    continue
                ret = dfs(v, f if f < cap[ei] else cap[ei])
                if ret > 0:
                    cap[ei] -= ret
                    cap[ei ^ 1] += ret
                    return ret
            return 0

        flow = 0
        while bfs():
            for i in range(n_nodes):
                it[i] = 0
            while True:
                pushed = dfs(s, 10**9)
                if pushed == 0:
                    break
                flow += pushed
                if flow >= 10**6:
                    return flow
        return flow

    # ---------- Vertex-disjoint-paths feasibility via super-node flow ----------
    # For a branch set B of size k, we need C(k,2) internally-disjoint paths.
    # Model: split each vertex v into v_in -> v_out (cap 1 for non-branches, cap INF for branches).
    # Each undirected edge (u,v) becomes two directed edges u_out->v_in and v_out->u_in (cap 1).
    # Add a source S and sink T. For each branch b, add S->b_in (cap 1) and b_out->T (cap 1).
    # We need flow = C(k,2). But wait: a standard max-flow gives us edge-disjoint paths from S to T,
    # each path goes S -> b1_in -> b1_out -> ... -> b2_in -> b2_out -> T. That's ONE path between a
    # pair of branches. We need ALL pairs. This single max-flow only gives us a set of paths that
    # collectively are vertex-disjoint, but they may not cover all pairs.
    #
    # Correct approach: For each pair (a,b) in B, we need a path. The vertex-disjointness is GLOBAL.
    # This is a "many-to-many" disjoint paths problem. A single max-flow from S to T with |B| sources
    # and |B| sinks (each branch both a source and sink) does NOT directly encode "every pair connected".
    #
    # Better: Use the fact that we need a topological minor of K_k. Equivalent to: find k internally-
    # disjoint paths covering all pairs. This is NP-hard in general, but for small k we can use
    # max-flow per pair with a vertex-capacity network, checking feasibility sequentially BUT using
    # max-flow (which finds the path that leaves the most residual capacity) rather than greedy BFS.
    #
    # Practical approach for k up to ~10: sequential assignment using max-flow for each pair,
    # but with a smarter ordering and backtracking. To keep it tractable, use a randomized
    # sequential max-flow with multiple restarts and a good pair ordering.

    def feasible_with_maxflow(branches):
        """Try to find internally-disjoint paths for all pairs using sequential max-flow.
        Returns (paths_dict, True) or (None, False)."""
        k = len(branches)
        bset = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        # Order pairs: most constrained first (fewest available interior vertices)
        # For simplicity, sort by distance (shortest path length) as a proxy
        # Actually, just try a fixed order and a few random orders
        for attempt in range(3):
            if attempt == 0:
                order = pairs
            else:
                order = pairs[:]
                random.shuffle(order)
            used_interior = set()
            paths = {}
            ok = True
            for (a, b) in order:
                banned = used_interior | bset
                banned.discard(a)
                banned.discard(b)
                path = _bfs_path(a, b, banned)
                if path is None:
                    ok = False
                    break
                interior = set(path[1:-1])
                if interior & used_interior or interior & bset:
                    ok = False
                    break
                used_interior |= interior
                paths[frozenset((a, b))] = path
            if ok:
                return paths, True
        return None, False

    def _bfs_path(u, v, banned):
        if u == v:
            return None
        if adj_bool[u][v]:
            return [u, v]
        parent = {u: None}
        q = deque([u])
        while q:
            cur = q.popleft()
            for nxt in adj_list[cur]:
                if nxt == v:
                    path = [v]
                    node = cur
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                if nxt in parent or nxt in banned:
                    continue
                parent[nxt] = cur
                q.append(nxt)
        return None

    # ---------- Branch-and-bound / iterative deepening over branch sets ----------
    # We'll use a randomized local search + simulated-annealing-like approach,
    # but the KEY difference is: we use a feasibility oracle that tries MULTIPLE
    # path orderings (not just one greedy order), and we do a proper local search
    # (add/remove/swap) with acceptance criteria.

    def validate(branches, paths):
        if not branches or len(branches) < 2:
            return 0
        bset = set(branches)
        if len(bset) != len(branches):
            return 0
        expected_pairs = set(frozenset((a, b)) for a, b in combinations(branches, 2))
        if len(paths) != len(expected_pairs):
            return 0
        used_interior = set()
        for fkey, path in paths.items():
            if len(path) < 2:
                return 0
            a, b = path[0], path[-1]
            if a not in bset or b not in bset:
                return 0
            if a == b:
                return 0
            if frozenset((a, b)) not in expected_pairs:
                return 0
            interior = set(path[1:-1])
            if interior & bset:
                return 0
            if interior & used_interior:
                return 0
            for i in range(len(path) - 1):
                if not adj_bool[path[i]][path[i + 1]]:
                    return 0
            used_interior |= interior
        return len(branches)

    best_count = 0
    best_result = None
    rng = random.Random(12345)

    def try_set(branches):
        nonlocal best_count, best_result
        if time_up():
            return
        paths, ok = feasible_with_maxflow(branches)
        if ok:
            cnt = validate(branches, paths)
            if cnt > best_count:
                best_count = cnt
                best_result = (sorted(branches), paths)

    # Phase 1: Try all small sets (k=2,3,4) to establish a baseline
    # k=2: all edges
    for u, v in edges:
        try_set([u, v])
    # k=3: all triangles
    es = set((min(u,v), max(u,v)) for u, v in edges)
    for a, b, c in combinations(range(n), 3):
        if (min(a,b),max(a,b)) in es and (min(b,c),max(b,c)) in es and (min(a,c),max(a,c)) in es:
            try_set([a, b, c])
            if time_up():
                break
    # k=4: sample some 4-sets
    all_vertices = list(range(n))
    # Try high-degree vertices first for k=4
    deg = [len(adj[i]) for i in range(n)]
    sorted_by_deg = sorted(range(n), key=lambda x: -deg[x])
    top = sorted_by_deg[:16]
    for quad in combinations(top, 4):
        try_set(list(quad))
        if time_up():
            break

    # Phase 2: Local search - add/remove/swap branches
    if best_result is not None:
        cur_branches = best_result[0][:]
        for iteration in range(500):
            if time_up():
                break
            # Try adding a random vertex
            candidates = [v for v in range(n) if v not in set(cur_branches)]
            if candidates:
                v_add = rng.choice(candidates)
                new_b = sorted(set(cur_branches) | {v_add})
                try_set(new_b)
                if best_result is not None and best_result[0] != cur_branches:
                    cur_branches = best_result[0][:]
            # Try removing a branch and adding another
            if len(cur_branches) > 2:
                v_rem = rng.choice(cur_branches)
                new_b = [b for b in cur_branches if b != v_rem]
                candidates = [v for v in range(n) if v not in set(new_b)]
                if candidates:
                    v_add = rng.choice(candidates)
                    new_b.append(v_add)
                    new_b.sort()
                    try_set(new_b)
                    if best_result is not None and best_result[0] != cur_branches:
                        cur_branches = best_result[0][:]
            # Try swapping two branches
            if len(cur_branches) >= 4:
                i, j = rng.sample(range(len(cur_branches)), 2)
                new_b = cur_branches[:]
                removed = new_b.pop(i)
                removed2 = new_b.pop(j)
                candidates = [v for v in range(n) if v not in set(new_b) and v != removed and v != removed2]
                if len(candidates) >= 2:
                    add1, add2 = rng.sample(candidates, 2)
                    new_b.extend([add1, add2])
                    new_b.sort()
                    try_set(new_b)
                    if best_result is not None and best_result[0] != cur_branches:
                        cur_branches = best_result[0][:]

    # Phase 3: Systematic expansion from best - try all single additions in order of degree
    if best_result is not None and not time_up():
        bset = set(best_result[0])
        candidates = [(deg[v], v) for v in range(n) if v not in bset]
        candidates.sort(reverse=True)
        for _, v in candidates:
            if time_up():
                break
            new_b = sorted(bset | {v})
            try_set(new_b)
            if best_result is not None:
                bset = set(best_result[0])
                candidates = [(deg[x], x) for x in range(n) if x not in bset]
                candidates.sort(reverse=True)

    # Phase 4: More random restarts with larger seeds
    for restart in range(300):
        if time_up():
            break
        size = rng.randint(3, min(7, n))
        seed = rng.sample(range(n), size)
        try_set(seed)

    # Final
    if best_result is None:
        u, v = edges[0]
        branches = [u, v]
        paths = {frozenset((u, v)): [u, v]}
    else:
        branches, paths = best_result

    branch_list = [int(b) for b in sorted(branches)]
    path_list = []
    for a, b in combinations(sorted(branches), 2):
        fkey = frozenset((a, b))
        p = paths.get(fkey)
        if p is None:
            p = [a, b]
        path_list.append([int(x) for x in p])

    return {'branches': branch_list, 'paths': path_list}


# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}
# EVOLVE-BLOCK-END
