"""Solver for large clique subdivisions in a pinned graph."""

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    import time
    from collections import deque
    from itertools import combinations

    t_start = time.time()
    DEADLINE = 82.0

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    adj_list = [list() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        adj_list[i] = sorted(adj[i])
    deg = [len(adj[i]) for i in range(n)]

    def time_up():
        return time.time() - t_start > DEADLINE

    # All-pairs shortest paths
    sp_dist = [[0] * n for _ in range(n)]
    for s in range(n):
        dist = [10**9] * n
        dist[s] = 0
        q = deque([s])
        while q:
            cur = q.popleft()
            for nb in adj_list[cur]:
                if dist[nb] > dist[cur] + 1:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        sp_dist[s] = dist

    # ---- Max-flow feasibility check on vertex-capacity network ----
    # For a candidate branch set B, build a network where:
    #   - Each non-branch vertex v is split into v_in -> v_out with capacity 1
    #   - Each branch vertex b is split into b_in -> b_out with capacity INF
    #   - Each original edge (u,v) becomes u_out -> v_in and v_out -> u_in with cap INF
    #   - For each pair (a,b) in B, we need 1 unit of flow from a_out to b_in
    #     and 1 unit from b_out to a_in  (or we can do a single super source/sink)
    #
    # Actually, the correct formulation: we need |B|*(|B|-1)/2 internally-disjoint
    # paths. Each non-branch vertex can be used at most once total.
    #
    # Simpler check: for each non-branch vertex, count how many pairs' shortest paths
    # go through it. But that's not exact.
    #
    # Better: use max-flow with a super source S and super sink T.
    # For each pair (a,b) in B, add edge S->a_in (cap 1) and b_out->T (cap 1).
    # But this doesn't enforce that each pair gets exactly one path.
    #
    # Actually the right model: we need to route |B|(|B|-1)/2 units of flow,
    # where each unit goes from one branch to another, and each non-branch
    # vertex has capacity 1.
    #
    # Let me use: super source S connects to each branch b_in with capacity (|B|-1).
    # Each branch b_out connects to super sink T with capacity (|B|-1).
    # Each non-branch v: v_in->v_out cap 1.
    # Each branch b: b_in->b_out cap (|B|-1).
    # Each edge (u,v): u_out->v_in cap 1, v_out->u_in cap 1.
    # Total flow needed: |B|*(|B|-1)/2.
    #
    # Wait, this allows a unit of flow from branch a to branch b to use vertex v,
    # and another unit from branch c to branch d to also use v. That's wrong.
    # Each non-branch vertex can only be on ONE path total.
    #
    # So: non-branch v_in->v_out cap 1. Branch b_in->b_out cap |B|-1.
    # S->b_in cap |B|-1 for each branch b. b_out->T cap |B|-1 for each branch b.
    # Edge (u,v): u_out->v_in cap 1 (if v is non-branch) or cap |B|-1 (if v is branch).
    #
    # Hmm, this is getting complex. Let me just use a simpler necessary condition:
    # For each non-branch vertex v, the number of branch pairs whose ALL paths
    # must go through v is at most 1. A simpler check: for each non-branch v,
    # count branches that are only reachable through v. If too many, infeasible.
    #
    # Actually, let me just use the vertex-connectivity check:
    # For each non-branch vertex v, if removing v disconnects more than 1 pair of
    # branches, it's infeasible. But that's also expensive.
    #
    # Let me implement a proper max-flow check.

    def max_flow_feasible(branches):
        """Check if a branch set is feasible using max-flow on vertex-capacity network.
        Returns True if |B|*(|B|-1)/2 internally-disjoint paths exist."""
        if time_up():
            return True  # skip check if time is up
        bset = set(branches)
        k = len(branches)
        if k < 2:
            return True
        needed = k * (k - 1) // 2
        INF = needed + 1

        # Node mapping: for each vertex v, create v_in and v_out
        # v_in = 2*v, v_out = 2*v + 1
        # S = 2*n, T = 2*n + 1
        S = 2 * n
        T = 2 * n + 1
        N = 2 * n + 2

        # Build adjacency for flow network
        # We'll use Edmonds-Karp (BFS-based) since N is small
        # adj_flow[u] = list of (v, capacity)
        cap = [[0] * N for _ in range(N)]

        # Vertex split edges
        for v in range(n):
            vin, vout = 2 * v, 2 * v + 1
            if v in bset:
                cap[vin][vout] = INF
            else:
                cap[vin][vout] = 1

        # Graph edges: u_out -> v_in and v_out -> u_in
        for u, v in edges:
            uout, vin = 2 * u + 1, 2 * v
            vout, uin = 2 * v + 1, 2 * u
            cap[uout][vin] = 1
            cap[vout][uin] = 1

        # Super source/sink
        for b in branches:
            bin_ = 2 * b
            bout = 2 * b + 1
            cap[S][bin_] = k - 1
            cap[bout][T] = k - 1

        # Edmonds-Karp
        flow = 0
        while True:
            if time_up():
                return True
            # BFS to find augmenting path
            parent = [-1] * N
            parent[S] = S
            q = deque([S])
            while q:
                u = q.popleft()
                for v in range(N):
                    if parent[v] == -1 and cap[u][v] > 0:
                        parent[v] = u
                        if v == T:
                            break
                        q.append(v)
                if parent[T] != -1:
                    break
            if parent[T] == -1:
                break
            # Find bottleneck
            path_flow = INF
            v = T
            while v != S:
                u = parent[v]
                path_flow = min(path_flow, cap[u][v])
                v = u
            # Update capacities
            v = T
            while v != S:
                u = parent[v]
                cap[u][v] -= path_flow
                cap[v][u] += path_flow
                v = u
            flow += path_flow
            if flow >= needed:
                return True
        return flow >= needed

    # ---- BFS path finding ----
    def bfs_path(src, dst, forbidden):
        if src == dst:
            return [src]
        parent = {src: -1}
        q = deque([src])
        while q:
            cur = q.popleft()
            if cur == dst:
                path = []
                node = dst
                while node != -1:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in adj_list[cur]:
                if nb not in parent:
                    if nb != dst and nb in forbidden:
                        continue
                    parent[nb] = cur
                    q.append(nb)
        return None

    def find_alternatives(u, v, used, bset, max_alts=5):
        alts = []
        p0 = bfs_path(u, v, used)
        if p0 is None:
            return alts
        alts.append(tuple(p0))
        seen = {tuple(p0)}
        for mid in p0[1:-1]:
            if len(alts) >= max_alts:
                break
            alt = bfs_path(u, v, used | {mid})
            if alt is not None:
                t = tuple(alt)
                if t not in seen:
                    seen.add(t)
                    alts.append(t)
        if len(alts) < max_alts:
            for a in p0[1:-1]:
                for b in p0[1:-1]:
                    if a == b:
                        continue
                    if len(alts) >= max_alts:
                        break
                    alt = bfs_path(u, v, used | {a, b})
                    if alt is not None:
                        t = tuple(alt)
                        if t not in seen:
                            seen.add(t)
                            alts.append(t)
        return alts

    def solve_paths(branches, order, max_alts=5):
        bset = set(branches)

        def bt(idx, used, paths):
            if time_up():
                return None
            if idx == len(order):
                return paths
            u, v = order[idx]
            alts = find_alternatives(u, v, used, bset, max_alts)
            if not alts:
                return None
            for path_t in alts:
                interior = set(path_t[1:-1])
                if interior & bset or interior & used:
                    continue
                new_used = used | interior
                paths[(min(u, v), max(u, v))] = list(path_t)
                result = bt(idx + 1, new_used, paths)
                if result is not None:
                    return result
                del paths[(min(u, v), max(u, v))]
            return None

        return bt(0, set(), {})

    def make_order(branches, mode):
        pairs = list(combinations(branches, 2))
        if mode == 0:
            pairs.sort(key=lambda p: (0 if p[1] in adj[p[0]] else 1,
                                       sp_dist[p[0]][p[1]], p[0], p[1]))
        elif mode == 1:
            pairs.sort(key=lambda p: -(deg[p[0]] + deg[p[1]]))
        elif mode == 2:
            pairs.sort(key=lambda p: (deg[p[0]], deg[p[1]]))
        elif mode == 3:
            pairs.sort(key=lambda p: (0 if p[1] in adj[p[0]] else 1,
                                       sp_dist[p[0]][p[1]], p[0], p[1]))
            pairs.reverse()
        elif mode == 4:
            pairs.sort(key=lambda p: (sp_dist[p[0]][p[1]], p[0], p[1]))
        elif mode == 5:
            # Sort by number of common neighbors (more connected pairs first)
            pairs.sort(key=lambda p: -len(adj[p[0]] & adj[p[1]]))
        else:
            pairs.sort(key=lambda p: (p[0], p[1]))
        return pairs

    def make_result(branches, paths):
        path_list = []
        for u, v in combinations(branches, 2):
            key = (min(u, v), max(u, v))
            path_list.append(list(paths[key]))
        return {'branches': list(branches), 'paths': path_list}

    def try_branch_set(branches, max_alts=5, use_flow_check=True):
        for mode in range(6):
            if time_up():
                return None
            order = make_order(branches, mode)
            paths = solve_paths(branches, order, max_alts)
            if paths is not None:
                return make_result(branches, paths)
        return None

    def try_branch_set_with_flow(branches, max_alts=5):
        """Use max-flow check first to quickly reject infeasible sets."""
        if not max_flow_feasible(branches):
            return None
        return try_branch_set(branches, max_alts, use_flow_check=False)

    # ---- Find cliques ----
    def find_cliques(max_size=8, limit=200):
        cliques = []
        candidates = sorted(range(n), key=lambda i: (-deg[i], i))
        for start in candidates:
            if time_up():
                break
            clique = [start]
            for v in candidates:
                if v == start:
                    continue
                if all(v in adj[w] for w in clique):
                    clique.append(v)
                if len(clique) >= max_size:
                    break
            if len(clique) >= 2:
                cliques.append(clique)
            if len(cliques) >= limit:
                break
        for start in range(n):
            if start in [c[0] for c in cliques]:
                continue
            clique = [start]
            for v in candidates:
                if v == start:
                    continue
                if all(v in adj[w] for w in clique):
                    clique.append(v)
                if len(clique) >= max_size:
                    break
            if len(clique) >= 2:
                cliques.append(clique)
            if len(cliques) >= limit * 2:
                break
        return cliques

    best = None
    best_size = 0

    # Phase 1: Greedy growth
    cliques = find_cliques(max_size=8, limit=100)
    cliques.sort(key=lambda c: (-len(c), -sum(deg[v] for v in c)))
    growth_pool = sorted(range(n), key=lambda i: (-deg[i], i))

    for seed in cliques:
        if time_up():
            break
        if len(seed) <= best_size:
            continue
        branches = list(seed)
        result = try_branch_set_with_flow(branches, max_alts=5)
        if result is None:
            for sz in range(len(seed) - 1, 1, -1):
                if sz <= best_size:
                    break
                for sub in combinations(seed, sz):
                    if time_up():
                        break
                    result = try_branch_set_with_flow(list(sub), max_alts=5)
                    if result is not None:
                        branches = list(sub)
                        break
                if result is not None:
                    break
            if result is None:
                continue
            best = result
            best_size = len(branches)

        changed = True
        while changed and not time_up():
            changed = False
            for cand in growth_pool:
                if cand in branches:
                    continue
                if len(branches) + 1 <= best_size:
                    break
                new_branches = branches + [cand]
                result = try_branch_set_with_flow(new_branches, max_alts=6)
                if result is not None:
                    branches = new_branches
                    best = result
                    best_size = len(branches)
                    changed = True
                    break

    # Phase 2: Top-down search
    if best_size < 10:
        candidates = sorted(range(n), key=lambda i: (-deg[i], i))
        for size in range(min(n, 14), best_size, -1):
            if time_up():
                break
            top_k = min(n, size + 8)
            pool = candidates[:top_k]
            tried = 0
            max_tries = 5000
            for combo in combinations(pool, size):
                if time_up() or tried >= max_tries:
                    break
                tried += 1
                branches = list(combo)
                if not max_flow_feasible(branches):
                    continue
                result = try_branch_set(branches, max_alts=6)
                if result is not None:
                    best = result
                    best_size = size
                    break
            if best_size >= size:
                break

    # Phase 3: Backtracking on promising sets
    if best_size < 8:
        for size in range(min(n, 10), max(best_size, 3), -1):
            if time_up():
                break
            if size <= best_size:
                break
            top_k = min(n, size + 6)
            pool = candidates[:top_k]
            tried = 0
            max_tries = 2000
            for combo in combinations(pool, size):
                if time_up() or tried >= max_tries:
                    break
                tried += 1
                branches = list(combo)
                if not max_flow_feasible(branches):
                    continue
                for mode in range(6):
                    if time_up():
                        break
                    order = make_order(branches, mode)
                    paths = solve_paths(branches, order, max_alts=7)
                    if paths is not None:
                        best = make_result(branches, paths)
                        best_size = size
                        break
                if best_size >= size:
                    break
            if best_size >= size:
                break

    # Fallback
    if best is None:
        for a, b, c in combinations(range(n), 3):
            if b in adj[a] and c in adj[a] and c in adj[b]:
                best = {
                    'branches': [a, b, c],
                    'paths': [[a, b], [a, c], [b, c]]
                }
                break
        if best is None:
            u, v = edges[0]
            best = {'branches': [u, v], 'paths': [[u, v]]}

    branches = [int(x) for x in best['branches']]
    paths = [[int(x) for x in p] for p in best['paths']]
    return {'branches': branches, 'paths': paths}
