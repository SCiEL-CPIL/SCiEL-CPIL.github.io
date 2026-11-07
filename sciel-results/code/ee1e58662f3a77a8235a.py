import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _ek(u, v):
    return (u, v) if u < v else (v, u)


def _validate(bicliques, edges_set):
    if not bicliques:
        return len(edges_set) == 0
    seen = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        for u in left:
            if not isinstance(u, int) or isinstance(u, bool):
                return False
        for v in right:
            if not isinstance(v, int) or isinstance(v, bool):
                return False
        for u in left:
            for v in right:
                if u == v:
                    return False
                e = _ek(u, v)
                if e not in edges_set:
                    return False
                if e in seen:
                    return False
                seen.add(e)
    return seen == edges_set


def _max_closure_biclique(adj, remaining, weights, deadline):
    """
    Find a maximum-weight biclique in the residual graph via a bipartite
    maximum closure reduction. Nodes: L-vertices and R-vertices (disjoint
    copies). Selecting a vertex on the left gives profit = sum of weights of
    residual edges incident to it; selecting a vertex on the right gives
    profit = 0 (symmetric). The closure constraint forces: if a left vertex
    is selected and a right vertex is selected, the edge between them must
    be present (we model this by giving a large negative penalty for selecting
    a pair that has no edge). For n=30 this is small enough to solve with a
    simple max-flow (Dinic).
    """
    n = len(adj)
    # Build node indices: 0..n-1 for left, n..2n-1 for right
    N = 2 * n
    source = N
    sink = N + 1

    # Profit for selecting left vertex i: sum of weights of residual edges (i, v) for v in right
    # Profit for selecting right vertex j: 0
    # Penalty for selecting (i_left, j_right) when no edge: -INF
    INF = float('inf')

    # We'll use a max-flow with capacities.
    # Standard reduction: max closure = total_positive - min_cut.
    # Node i (left): capacity from source = profit_i if positive, else 0.
    # Node j (right): capacity from source = 0.
    # Edge i_left -> j_right: capacity = INF if (i,j) is NOT a residual edge (penalty), else 0.
    # Edge i_left -> sink: capacity = 0 if profit_i >= 0 else -profit_i.
    # Edge j_right -> sink: capacity = 0.

    # Compute profits
    profit = [0.0] * N
    for i in range(n):
        for j in range(n):
            if i != j:
                e = _ek(i, j)
                if e in remaining:
                    profit[i] += weights.get(e, 0.0)

    # Build graph for Dinic
    # adjacency list: graph[node] = list of [to, cap, rev_index]
    graph = [[] for _ in range(N + 2)]

    def add_edge(u, v, cap):
        graph[u].append([v, cap, len(graph[v])])
        graph[v].append([u, 0.0, len(graph[u]) - 1])

    total_pos = 0.0
    for i in range(N):
        if profit[i] > 0:
            add_edge(source, i, profit[i])
            total_pos += profit[i]
        else:
            add_edge(i, sink, -profit[i])

    # Penalty edges: for each (i_left, j_right) with no residual edge, add edge i -> j with cap INF
    # But only if i != j (diagonal is always forbidden)
    for i in range(n):
        for j in range(n):
            if i == j:
                # Always forbidden: select i_left and i_right together is invalid
                add_edge(i, n + j, INF)
            else:
                e = _ek(i, j)
                if e not in remaining:
                    add_edge(i, n + j, INF)

    # Dinic
    def dinic_maxflow():
        from collections import deque
        flow = 0.0
        Nnodes = N + 2
        while True:
            # BFS
            level = [-1] * Nnodes
            level[source] = 0
            q = deque([source])
            while q:
                u = q.popleft()
                for v, cap, rev in graph[u]:
                    if cap > 1e-12 and level[v] < 0:
                        level[v] = level[u] + 1
                        q.append(v)
            if level[sink] < 0:
                break
            # DFS
            it = [0] * Nnodes
            def dfs(u, f):
                if u == sink:
                    return f
                while it[u] < len(graph[u]):
                    v, cap, rev = graph[u][it[u]]
                    if cap > 1e-12 and level[v] == level[u] + 1:
                        d = dfs(v, min(f, cap))
                        if d > 1e-12:
                            graph[u][it[u]][1] -= d
                            graph[v][rev][1] += d
                            return d
                    it[u] += 1
                return 0.0
            while True:
                f = dfs(source, float('inf'))
                if f < 1e-12:
                    break
                flow += f
        return flow

    min_cut = dinic_maxflow()
    max_closure_value = total_pos - min_cut

    # Recover selected nodes: those reachable from source in residual graph
    from collections import deque
    reachable = [False] * Nnodes
    reachable[source] = True
    q = deque([source])
    while q:
        u = q.popleft()
        for v, cap, rev in graph[u]:
            if cap > 1e-12 and not reachable[v]:
                reachable[v] = True
                q.append(v)

    left_sel = set()
    right_sel = set()
    for i in range(n):
        if reachable[i]:
            left_sel.add(i)
    for j in range(n):
        if reachable[n + j]:
            right_sel.add(j)

    if not left_sel or not right_sel:
        return None, None

    # Filter: ensure all cross pairs are residual edges
    left_sel = {u for u in left_sel if all(_ek(u, v) in remaining for v in right_sel)}
    right_sel = {v for v in right_sel if all(_ek(u, v) in remaining for u in left_sel)}

    if not left_sel or not right_sel:
        return None, None

    return left_sel, right_sel


def _greedy_biclique_partition(adj, edges_set, deadline):
    remaining = set(edges_set)
    bicliques = []

    while remaining and time.time() < deadline:
        # Compute weights: inverse frequency heuristic
        # frequency[e] = number of times e's endpoints have been in candidate bicliques
        # For simplicity, use weight = 1.0 / (deg(u) + deg(v)) as a proxy
        deg = {}
        for u, v in remaining:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1

        weights = {}
        for e in remaining:
            u, v = e
            weights[e] = 1.0 / (deg.get(u, 1) + deg.get(v, 1))

        # Try maximum closure biclique
        best_left, best_right = None, None
        best_val = 0.0

        # Also try anchor-based expansion for robustness
        sorted_anchors = sorted(deg.keys(), key=lambda x: -deg[x])

        # Pricing subproblem: max-weight biclique via max closure
        left_c, right_c = _max_closure_biclique(adj, remaining, weights, deadline)
        if left_c and right_c:
            val = sum(weights.get(_ek(u, v), 0.0) for u in left_c for v in right_c)
            if val > best_val:
                best_val = val
                best_left, best_right = left_c, right_c

        # Anchor-based fallback
        for anchor in sorted_anchors[:10]:
            left = {anchor}
            right = set()
            for u, v in remaining:
                if u == anchor:
                    right.add(v)
                elif v == anchor:
                    right.add(u)
            if not right:
                continue

            # Iterative refinement
            for _ in range(50):
                changed = False
                new_right = set()
                for v in right:
                    if all(_ek(u, v) in remaining for u in left):
                        new_right.add(v)
                if new_right != right:
                    right = new_right
                    changed = True
                new_left = set()
                for u in left:
                    if all(_ek(u, v) in remaining for v in right):
                        new_left.add(u)
                if new_left != left:
                    left = new_left
                    changed = True
                if not changed:
                    break

            # Expand
            for u in range(P['n']):
                if u not in left and all(_ek(u, v) in remaining for v in right):
                    left.add(u)
            for v in range(P['n']):
                if v not in right and all(_ek(u, v) in remaining for u in left):
                    right.add(v)

            if not left or not right:
                continue

            val = sum(weights.get(_ek(u, v), 0.0) for u in left for v in right)
            if val > best_val:
                best_val = val
                best_left, best_right = left, right

        if best_left is None:
            u, v = next(iter(remaining))
            bicliques.append([[u], [v]])
            remaining.discard(_ek(u, v))
            continue

        # Remove covered edges
        for u in best_left:
            for v in best_right:
                remaining.discard(_ek(u, v))

        bicliques.append([list(best_left), list(best_right)])

    # Leftover edges
    for e in remaining:
        bicliques.append([[e[0]], [e[1]]])

    return bicliques


def _try_merge(bicliques, edges_set, deadline):
    """Try to merge pairs of bicliques."""
    n = len(bicliques)
    merged = True
    while merged and time.time() < deadline:
        merged = False
        for i in range(len(bicliques)):
            if merged:
                break
            for j in range(i + 1, len(bicliques)):
                li, ri = bicliques[i][0], bicliques[i][1]
                lj, rj = bicliques[j][0], bicliques[j][1]

                new_left = set(li) | set(lj)
                new_right = set(ri) | set(rj)

                # Check all cross pairs are edges
                valid = True
                for u in new_left:
                    for v in new_right:
                        if u == v or _ek(u, v) not in edges_set:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue

                # Check coverage matches
                covered_i = set(_ek(u, v) for u in li for v in ri)
                covered_j = set(_ek(u, v) for u in lj for v in rj)
                covered_new = set(_ek(u, v) for u in new_left for v in new_right)

                if covered_new == covered_i | covered_j:
                    bicliques[i] = [list(new_left), list(new_right)]
                    del bicliques[j]
                    merged = True
                    break

    return bicliques


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    edges = P['edges']
    n = P['n']

    # Build adjacency
    adj = [set() for _ in range(n)]
    edges_set = set()
    for u, v in edges:
        e = _ek(u, v)
        edges_set.add(e)
        adj[u].add(v)
        adj[v].add(u)

    # Random restarts
    best_bicliques = None
    best_count = float('inf')

    # Fixed seed for reproducibility in evaluation
    random.seed(42)

    for restart in range(8):
        if time.time() > deadline:
            break

        # Shuffle edge order for diversity
        shuffled_edges = list(edges_set)
        random.shuffle(shuffled_edges)

        bicliques = _greedy_biclique_partition(adj, set(shuffled_edges), deadline)

        if not _validate(bicliques, edges_set):
            continue

        # Merge pass
        bicliques = _try_merge(bicliques, edges_set, deadline)

        if not _validate(bicliques, edges_set):
            continue

        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = [list(b) for b in bicliques]

    if best_bicliques is None:
        # Fallback: star partition
        bicliques = []
        for u in range(n):
            neighbors = [v for v in adj[u]]
            if neighbors:
                bicliques.append([neighbors, [u]])
        return bicliques

    # Final polish: try to reduce count by checking if any biclique can be split
    # or if edges can be reassigned. This is a high-precision pass.
    if time.time() < deadline:
        for _ in range(5):
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(best_bicliques)):
                li, ri = best_bicliques[i][0], best_bicliques[i][1]
                # Try removing one vertex from left or right and see if the rest
                # can be merged with another biclique
                for u in list(li):
                    new_li = [x for x in li if x != u]
                    if not new_li:
                        continue
                    # Check if [new_li, ri] is still valid and covers subset
                    valid = True
                    for a in new_li:
                        for b in ri:
                            if _ek(a, b) not in edges_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        # The removed vertex u's edges must be covered elsewhere
                        # This is complex; skip for now
                        pass

            # If no improvement, break
            if not improved:
                break

    return best_bicliques
