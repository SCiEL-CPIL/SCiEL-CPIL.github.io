import time
import random
import sys
from collections import deque, defaultdict

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    t0 = time.time()
    DEADLINE = 85.0
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def time_up():
        return (time.time() - t0) > DEADLINE

    # ---------- Dinic max-flow ----------
    class Dinic:
        def __init__(self, N):
            self.N = N
            self.g = [[] for _ in range(N)]
        def add_edge(self, u, v, cap):
            self.g[u].append([v, cap, len(self.g[v])])
            self.g[v].append([u, 0, len(self.g[u]) - 1])
        def max_flow(self, s, t):
            N = self.N
            flow = 0
            INF = 10**9
            level = [0] * N
            it = [0] * N
            from collections import deque as _dq
            while True:
                level = [-1] * N
                level[s] = 0
                q = _dq([s])
                while q:
                    u = q.popleft()
                    for e in self.g[u]:
                        if e[1] > 0 and level[e[0]] < 0:
                            level[e[0]] = level[u] + 1
                            q.append(e[0])
                if level[t] < 0:
                    return flow
                it = [0] * N
                def dfs(u, f):
                    if u == t:
                        return f
                    while it[u] < len(self.g[u]):
                        e = self.g[u][it[u]]
                        if e[1] > 0 and level[u] < level[e[0]]:
                            d = dfs(e[0], min(f, e[1]))
                            if d > 0:
                                e[1] -= d
                                self.g[e[0]][e[2]][1] += d
                                return d
                        it[u] += 1
                    return 0
                while True:
                    f = dfs(s, INF)
                    if f == 0:
                        break
                    flow += f
            return flow

    # ---------- Max-flow feasibility oracle for a branch set ----------
    # For candidate B: build split graph.
    #   v_in = 2*v, v_out = 2*v+1
    #   S = 2n, T = 2n+1
    # Arcs:
    #   S -> v_in for each v in B (cap 1)
    #   v_out -> T for each v in B (cap 1)
    #   v_in -> v_out: cap 1 if v not in B, cap INF if v in B
    #   u_out -> v_in for each directed edge (u,v) (cap 1)
    # We need flow = |B| (each branch sends 1 unit to exactly one other branch).
    # Then decompose: each unit path S->...->T corresponds to a branch pair.
    # The path in original graph: for each branch a, trace the flow from S->a_in ... to b_out->T.
    #   a_in -> ... -> b_out: the sequence of v_in->v_out and u_out->v_in arcs.
    #   Original path: a, (interior vertices in order), b.
    #   We need to extract the actual vertex sequence.

    def flow_oracle(B):
        """Return list of paths (one per branch, each path is [branch_a, ..., branch_b]) or None."""
        kB = len(B)
        if kB < 2:
            return None
        Bset = set(B)
        Nnodes = 2 * n + 2
        S = 2 * n
        T = 2 * n + 1
        INF_CAP = 1000  # effectively infinite for our purposes

        din = Dinic(Nnodes)
        # S -> v_in
        for v in B:
            din.add_edge(S, 2 * v, 1)
        # v_out -> T
        for v in B:
            din.add_edge(2 * v + 1, T, 1)
        # v_in -> v_out
        for v in range(n):
            cap = INF_CAP if v in Bset else 1
            din.add_edge(2 * v, 2 * v + 1, cap)
        # u_out -> v_in for each directed edge
        for u, v in edges:
            din.add_edge(2 * u + 1, 2 * v, 1)
            din.add_edge(2 * v + 1, 2 * u, 1)

        flow = din.max_flow(S, T)
        if flow < kB:
            return None

        # Decompose: find kB paths from S to T in the residual (used) arcs.
        # For each branch a in B, the path starts S -> 2a. Follow used arcs.
        # We need to track which arcs carry flow. Since we used Dinic, we can
        # extract paths by walking from S along arcs with positive flow in the
        # original direction (i.e., the forward arc has residual < original cap,
        # or equivalently the reverse arc has residual > 0).
        # Simpler: rebuild a "flow graph" by checking each added edge's flow.
        # Actually, let's just do a DFS on the residual to find S->T paths.
        # Each S->T path in the flow network corresponds to one branch pair.

        # Build flow graph: for each arc (u,v) with flow>0, add directed edge u->v.
        # We'll track flow by looking at the residual: for each edge in din.g[u],
        # if the reverse edge (in din.g[v]) has residual > 0, then flow on u->v is that.
        # This is messy. Let's instead just do a BFS/DFS from S in the residual
        # network to find kB vertex-disjoint (in terms of S->T paths) paths.
        # Actually, since the flow is integral and each S->v_in and v_out->T has cap 1,
        # the kB paths are naturally separated by which branch they start/end at.

        # For each branch a, trace: S -> 2a (must have flow 1).
        # Then from 2a, follow the unique outgoing arc with flow > 0 (since in-capacity is 1).
        # Continue until we reach some 2b+1 -> T.
        # The sequence of nodes gives us the path.

        paths = []
        for a in B:
            # Trace from node 2a
            node = 2 * a  # a_in
            seq = [a]  # start at branch a
            # From a_in, the flow goes to a_out (the v_in->v_out arc)
            # Then from a_out, it goes to some v_in via an edge arc.
            # We need to find the next node.
            # In the flow network, from 2a (a_in), the only outgoing arc is to 2a+1 (a_out).
            # From 2a+1 (a_out), outgoing arcs go to 2v for v in adj[a] (edge arcs) and to T (if a in B).
            # But the flow from a_out should go to some v_in (not to T, since that would be a self-path).
            # Actually, a_out -> T has cap 1 and is only used if a's flow ends at a (self), which shouldn't happen.
            # Let's just follow the flow.

            # Find which outgoing arc from current node has flow > 0.
            # For node 2a: outgoing is only to 2a+1.
            # For node 2a+1: outgoing arcs to 2v (v in adj[a]) and to T.
            # We need to check which one has flow.

            # Let's write a helper: given a node, find the next node in the flow.
            def next_node_in_flow(u):
                # Check all edges from u in din.g[u]
                for e in din.g[u]:
                    v, cap, rev_idx = e
                    # The reverse edge is in din.g[v][rev_idx]
                    rev_cap = din.g[v][rev_idx][1]
                    if rev_cap > 0:
                        # Flow of rev_cap on u->v
                        return v, rev_cap
                return None, 0

            # Start tracing
            cur = 2 * a  # a_in
            # First step: a_in -> a_out
            cur = 2 * a + 1  # a_out
            while True:
                nxt, f = next_node_in_flow(cur)
                if nxt is None:
                    return None  # shouldn't happen
                if nxt == T:
                    # We've reached T. The previous node is 2b+1 for some b.
                    # cur is 2b+1, so b = cur // 2
                    b = cur // 2
                    seq.append(b)
                    break
                elif nxt % 2 == 0:
                    # nxt is v_in for v = nxt // 2
                    v = nxt // 2
                    seq.append(v)
                    cur = nxt
                    # Next: v_in -> v_out
                    cur = nxt + 1
                else:
                    # nxt is v_out, which shouldn't happen directly from v_in
                    return None
            paths.append(seq)

        # Now we have kB paths, each [a, ..., b]. We need to form the |B| choose 2 unordered-pair paths.
        # Each branch a appears as a start in exactly one path and as an end in exactly one path.
        # The path starting at a and ending at b represents the connection between a and b.
        # But we need paths for ALL pairs, not just a matching.
        # Wait: the flow formulation gives a set of kB paths that form a 2-regular structure (each branch has in-degree 1 and out-degree 1 in the flow), i.e., a collection of cycles covering all branches. This is NOT the same as having a path for every pair!

        # I made a fundamental error: the max-flow with |B| units gives a set of |B| paths that form a permutation/cycles, NOT a complete graph of paths. The clique subdivision requires C(|B|,2) paths, one for each pair.

        # This means the max-flow oracle as formulated does NOT verify the clique subdivision property. It only verifies that the branches can be arranged in a cycle (or set of cycles) with disjoint interiors.

        # So this approach is fundamentally flawed for the actual problem. Let me fall back to the heuristic.
        return None

    # Since the max-flow oracle is flawed for the actual problem (it checks for a cycle cover, not a complete graph of paths),
    # I'll use the heuristic approach but with a different structural idea:
    # Use a "greedy with backtracking" that tries to build the branch set incrementally,
    # using a smarter path-finding that considers the global packing constraint.

    # Actually, let me try a different structural approach:
    # The key insight is that for a clique subdivision, the union of all path interiors must be a set of
    # vertex-disjoint paths. This is equivalent to saying that if we remove all branch vertices,
    # the remaining graph must contain a set of vertex-disjoint paths that connect all pairs of branches.
    #
    # A different approach: use the fact that the graph is small (24 vertices).
    # Try all subsets of size k (from high to low) and check feasibility with a more careful method.
    # For k up to 8, C(24,8) = 735471, which is feasible if the check is fast.

    # For a given branch set B, check if clique subdivision exists:
    # This is the "disjoint paths problem" for a complete graph on |B| terminals, which is NP-hard in general
    # but for small |B| and small n, we can use a search.

    # For |B| <= 4, we can enumerate all ways to route the paths.
    # For larger |B|, use a heuristic.

    # Let me try a different structural approach: use the fact that the graph is small and use
    # a "color-coding" or "algebraic" method? No, that's overkill.

    # Instead, let me use a simple but effective approach:
    # 1. Find a large clique (as before).
    # 2. Try to grow it by adding vertices, using a more careful path-packing check.
    # 3. The key improvement: when checking if a branch set B is feasible, use a backtracking search
    #    that tries all orderings of the pairs, not just BFS.

    # Actually, for this specific graph, let me just use a more aggressive search.

    # Let me try: enumerate all subsets of size 6, 7, 8 and check feasibility with a fast heuristic.
    # C(24,6) = 134596, C(24,7) = 346104, C(24,8) = 735471. Total ~1.2M subsets.
    # If each check takes ~1ms, that's ~1200s, too slow.
    # If each check takes ~0.1ms, that's ~120s, still too slow.
    # So we need a faster check or fewer subsets.

    # Let me use a different approach: start from the maximum clique and try to grow it.

    def find_max_clique():
        best = []
        cands = sorted(range(n), key=lambda x: -len(adj[x]))
        def expand(clique, cand):
            nonlocal best
            if len(clique) > len(best):
                best = list(clique)
            if not cand:
                return
            for i, v in enumerate(cand):
                expand(clique + [v], [u for u in cand[i + 1:] if u in adj[v]])
        expand([], cands)
        return best

    clique = find_max_clique()

    def bfs_path(s, t, banned):
        if s == t:
            return None
        if t in adj[s]:
            return [s, t]
        parent = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v == s or v in parent or v in banned:
                    continue
                if v == t:
                    path = [v]
                    cur = u
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    return path
                parent[v] = u
                q.append(v)
        return None

    def find_paths(branches, order=None):
        m = len(branches)
        if m < 2:
            return None
        used = set()
        paths = {}
        if order is None:
            pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
            pairs.sort(key=lambda p: (0 if branches[p[1]] in adj[branches[p[0]]] else 1,
                                      -len(adj[branches[p[0]]]), -len(adj[branches[p[1]]])))
        else:
            pairs = order
        for (i, j) in pairs:
            a, b = branches[i], branches[j]
            banned = set(used)
            banned.update(branches)
            banned.discard(a)
            banned.discard(b)
            path = bfs_path(a, b, banned)
            if path is None:
                return None
            used.update(path[1:-1])
            paths[(i, j)] = path
        return [paths[p] for p in pairs]

    def find_paths_multi(branches, k=8):
        p = find_paths(branches)
        if p is not None:
            return p
        for _ in range(k):
            pairs = [(i, j) for i in range(len(branches)) for j in range(i + 1, len(branches))]
            random.shuffle(pairs)
            p = find_paths(branches, pairs)
            if p is not None:
                return p
        return None

    heur_branches = list(clique)
    heur_paths = find_paths_multi(clique, 10)
    if heur_paths is None and len(clique) >= 2:
        for sz in range(len(clique) - 1, 1, -1):
            sub = clique[:sz]
            heur_paths = find_paths_multi(sub, 8)
            if heur_paths is not None:
                heur_branches = sub
                break
    if heur_paths is None:
        u, v = edges[0]
        heur_branches = [u, v]
        heur_paths = [[u, v]]

    best_branches = heur_branches
    best_paths = heur_paths
    best_count = len(heur_branches)

    def update(b, p):
        nonlocal best_branches, best_paths, best_count
        if p is not None and len(b) > best_count:
            best_branches, best_paths, best_count = list(b), list(p), len(b)

    # Grow from clique
    if not time_up():
        cur = list(clique)
        cur_paths = find_paths_multi(cur, 10)
        if cur_paths is not None:
            update(cur, cur_paths)
        changed = True
        while changed and not time_up():
            changed = False
            candidates = [v for v in range(n) if v not in cur]
            candidates.sort(key=lambda v: len(adj[v]))
            for v in candidates:
                if time_up():
                    break
                trial = cur + [v]
                p = find_paths_multi(trial, 8)
                if p is not None:
                    cur = trial
                    changed = True
                    update(cur, p)
                    break

    # Local search
    if not time_up():
        improved = True
        while improved and not time_up():
            improved = False
            cur_b = list(best_branches)
            cur_set = set(cur_b)
            for idx in range(len(cur_b)):
                if time_up():
                    break
                for v in range(n):
                    if v in cur_set:
                        continue
                    new_b = cur_b[:idx] + cur_b[idx + 1:] + [v]
                    p = find_paths_multi(new_b, 8)
                    if p is not None and len(new_b) > best_count:
                        update(new_b, p)
                        improved = True
                        break
                if improved:
                    break

    # Random restarts
    if not time_up():
        for _ in range(300):
            if time_up():
                break
            sz = min(n, max(2, best_count + random.randint(-1, 2)))
            seed = random.sample(range(n), sz)
            cur = list(seed)
            p = find_paths_multi(cur, 6)
            if p is not None:
                update(cur, p)
            changed = True
            while changed and not time_up():
                changed = False
                for v in range(n):
                    if v in cur:
                        continue
                    trial = cur + [v]
                    p = find_paths_multi(trial, 5)
                    if p is not None:
                        cur = trial
                        changed = True
                        update(cur, p)
                        break

    if best_paths is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = [[u, v]]

    branches_out = [int(b) for b in best_branches]
    paths_out = [[int(x) for x in p] for p in best_paths]
    return {'branches': branches_out, 'paths': paths_out}
