import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    def remaining():
        return deadline - time.time()

    # ---- C4-safe edge-add check via bitmasks ----
    # Adding edge (u,v) creates a C4 iff there exist a,b with u~a, a~b, b~v, a!=b.
    # This is equivalent to N(u) & N(v) != 0 (a vertex counts as its own neighbour
    # because we store the adjacency mask including the vertex bit), which captures
    # every 4-cycle containing the new edge.
    def c4_safe(adj_mask, u, v):
        return (adj_mask[u] & adj_mask[v]) == 0

    def compute_alpha(adj_mask, n):
        """Exact maximum independent set size via branch-and-bound on bitmasks."""
        best = [0]

        def dfs(candidates, size):
            if size > best[0]:
                best[0] = size
            if candidates == 0:
                return
            if size + candidates.bit_count() <= best[0]:
                return
            # pick vertex with max degree inside candidate subgraph
            c = candidates
            max_deg = -1
            chosen = -1
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj_mask[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    chosen = v
            if chosen == -1:
                return
            # include chosen
            dfs(candidates & ~(adj_mask[chosen] | (1 << chosen)), size + 1)
            # exclude chosen
            dfs(candidates & ~(1 << chosen), size)

        dfs((1 << n) - 1, 0)
        return best[0]

    def has_alpha_at_least(adj_mask, n, k):
        """True if there is an independent set of size >= k."""
        found = [False]

        def dfs(candidates, size):
            if found[0]:
                return
            if size >= k:
                found[0] = True
                return
            if candidates == 0:
                return
            if size + candidates.bit_count() < k:
                return
            c = candidates
            max_deg = -1
            chosen = -1
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj_mask[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    chosen = v
            if chosen == -1:
                return
            dfs(candidates & ~(adj_mask[chosen] | (1 << chosen)), size + 1)
            if not found[0]:
                dfs(candidates & ~(1 << chosen), size)

        dfs((1 << n) - 1, 0)
        return found[0]

    def add_edge(adj_mask, u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    def local_search(adj_mask, n, max_steps=30000):
        """Greedily add C4-safe edges that most reduce the independence number."""
        steps = 0
        while steps < max_steps and remaining() > 8.0:
            steps += 1
            # candidate non-edges that are C4-safe
            best_gain = 0
            best_pair = None
            for u in range(n):
                um = adj_mask[u]
                for v in range(u + 1, n):
                    if (um >> v) & 1:
                        continue
                    if (adj_mask[u] & adj_mask[v]) != 0:
                        continue
                    # gain ~ number of independent-set candidates destroyed
                    # = common-neighbours of u and v (both lose them as candidates)
                    gain = (adj_mask[u] & adj_mask[v]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_pair = (u, v)
            if best_pair is None:
                break
            u, v = best_pair
            add_edge(adj_mask, u, v)
        return adj_mask

    def polish(adj_mask, n):
        """After alpha<11, keep adding any C4-safe edge to harden the graph."""
        while remaining() > 3.0:
            added = False
            for u in range(n):
                um = adj_mask[u]
                for v in range(u + 1, n):
                    if (um >> v) & 1:
                        continue
                    if (adj_mask[u] & adj_mask[v]) == 0:
                        add_edge(adj_mask, u, v)
                        added = True
                        um = adj_mask[u]
            if not added:
                break
        return adj_mask

    # ---- base structures ----
    def hs_graph(n=50):
        """(50,7,0,1) strongly regular C4-free graph (Hoffman-Singleton)."""
        adj = [0] * n
        for i in range(n):
            for d in (1, 2, 3, 4, 10, 14, 15):
                j = (i + d) % n
                if j > i:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def disjoint_triangles(n):
        """10 disjoint triangles (n=30, alpha=10), guaranteed C4-free."""
        adj = [0] * n
        for t in range(n // 3):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            add_edge(adj, a, b)
            add_edge(adj, a, c)
            add_edge(adj, b, c)
        return adj

    def to_result(adj_mask, n):
        edges = []
        for u in range(n):
            um = adj_mask[u]
            for v in range(u + 1, n):
                if (um >> v) & 1:
                    edges.append([u, v])
        return {'n': n, 'edges': edges}

    # ---- fallbacks ----
    fb_n = 30
    fb_adj = disjoint_triangles(fb_n)
    best_result = to_result(fb_adj, fb_n)
    best_n = fb_n

    # ---- main search: try larger n, refine via local search ----
    for n in (43, 42, 41, 40, 39):
        if remaining() < 20.0:
            break
        # multiple randomised local-search attempts
        for attempt in range(4):
            if remaining() < 20.0:
                break
            # start from the HS base on n vertices (first n of the 50-vertex graph)
            adj = hs_graph(n)
            # deterministic local search to shrink alpha
            local_search(adj, n)
            if remaining() < 6.0:
                break
            if not has_alpha_at_least(adj, n, 11):
                # success: alpha <= 10
                polish(adj, n)
                # final verification
                if not has_alpha_at_least(adj, n, 11):
                    res = to_result(adj, n)
                    if n > best_n:
                        best_n = n
                        best_result = res
                    return res
        # if we get here without success for this n, continue to smaller n
        if remaining() < 20.0:
            break

    return best_result
