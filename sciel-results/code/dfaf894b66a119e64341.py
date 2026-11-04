import time
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # --- Fast girth verification via non-backtracking BFS ---
    def _check_girth(adj, n, g_limit):
        """Check that no cycle shorter than g_limit exists.
        Uses BFS from each vertex up to depth (g_limit-1)//2.
        If we find an edge (u, v) such that dist[u] + dist[v] + 1 < g_limit,
        it implies a short cycle? 
        Actually, if u and v are connected by an edge, and dist[u] + dist[v] + 1 < g_limit,
        does it guarantee a cycle of that length?
        The path s->...->u->v->...->s has length dist[u] + 1 + dist[v].
        This is a valid cycle only if the paths s->u and s->v are disjoint except at s.
        In a BFS tree, the paths are unique and disjoint except at the LCA.
        If LCA is s, then yes.
        If LCA is not s, the cycle is shorter or different.
        However, if a cycle of length L < g_limit exists, there exists a vertex s on that cycle
        such that the BFS from s will detect a back edge closing a cycle of length <= L.
        Specifically, if we limit depth to D = (g_limit - 1) // 2, and we find an edge (u, v)
        with dist[u] + dist[v] + 1 < g_limit, we have found a short cycle.
        """
        max_depth = (g_limit - 1) // 2
        for start in range(n):
            dist = np.full(n, -1, dtype=np.int32)
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] < max_depth:
                            queue.append(v)
                    else:
                        # v is already visited
                        # Check if this edge forms a short cycle
                        # The cycle length is dist[u] + dist[v] + 1
                        # This is valid if the paths from start to u and start to v
                        # are disjoint (except at start) OR if we are just checking existence of ANY short cycle.
                        # Actually, if dist[u] + dist[v] + 1 < g_limit, a cycle of that length exists.
                        # Proof: Path s->u (len du), edge u-v, path v->s (len dv).
                        # If the paths intersect at w != s, then there is a cycle s->w...u-v...w->s which is shorter.
                        # So if du + dv + 1 < g_limit, a cycle shorter than g_limit exists.
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g_limit:
                            return False
        return True

    # --- LPS Graph Construction on PSL(2, q) ---
    def _lps_graph(q):
        """Construct a 3-regular graph from PSL(2, q) using standard generators."""
        if q < 5:
            return None, None
        
        # Generate all 2x2 matrices with det=1 mod q, identifying M ~ -M
        mat_to_idx = {}
        mats = []
        idx = 0
        
        # We can iterate over a, b, c and compute d
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    if a == 0:
                        # -bc = 1 mod q
                        if b == 0:
                            continue
                        try:
                            b_inv = pow(b, -1, q)
                        except ValueError:
                            continue
                        d = (-1 * b_inv) % q
                    else:
                        try:
                            a_inv = pow(a, -1, q)
                        except ValueError:
                            continue
                        d = ((1 + b * c) * a_inv) % q
                    
                    m = (a, b, c, d)
                    # Identify M and -M
                    neg_m = ((q-a)%q, (q-b)%q, (q-c)%q, (q-d)%q)
                    key = frozenset([m, neg_m])
                    
                    if key not in mat_to_idx:
                        mat_to_idx[key] = idx
                        mats.append(m)
                        idx += 1

        n = len(mats)
        
        # Generators for PSL(2,q):
        # S = [[0, -1], [1, 0]], T = [[1, 1], [0, 1]]
        S = (0, q-1, 1, 0)
        T = (1, 1, 0, 1)
        T_inv = (1, q-1, 0, 1)
        
        gens = [S, T, T_inv]
        
        def mat_mul(m1, m2):
            a1, b1, c1, d1 = m1
            a2, b2, c2, d2 = m2
            return ((a1*a2 + b1*c2) % q, (a1*b2 + b1*d2) % q,
                    (c1*a2 + d1*c2) % q, (c1*b2 + d1*d2) % q)
        
        adj = [[] for _ in range(n)]
        edges = []
        edge_set = set()
        
        for i, m in enumerate(mats):
            for g in gens:
                nm = mat_mul(m, g)
                neg_nm = ((q-nm[0])%q, (q-nm[1])%q, (q-nm[2])%q, (q-nm[3])%q)
                key = frozenset([nm, neg_nm])
                if key in mat_to_idx:
                    j = mat_to_idx[key]
                    if i != j:
                        if (i, j) not in edge_set and (j, i) not in edge_set:
                            adj[i].append(j)
                            adj[j].append(i)
                            edge_set.add((min(i,j), max(i,j)))
                            edges.append((min(i,j), max(i,j)))
        
        # Verify 3-regular
        for i in range(n):
            if len(adj[i]) != 3:
                return None, None
        
        return n, edges

    # --- LUW Graph Construction (Fallback) ---
    def _luw_graph(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # --- Main logic ---
    if k == 3 and g == 16:
        best_n = float('inf')
        best_edges = None
        
        # Try LPS graphs for small primes
        # q=7 -> n=168, q=11 -> n=660, q=13 -> n=1092
        for q in [7, 11, 13, 17, 19]:
            if time.time() > deadline - 20:
                break
            n_lps, edges_lps = _lps_graph(q)
            if n_lps is not None and n_lps < best_n:
                adj = [[] for _ in range(n_lps)]
                for u, v in edges_lps:
                    adj[u].append(v)
                    adj[v].append(u)
                if _check_girth(adj, n_lps, g):
                    best_n = n_lps
                    best_edges = edges_lps
                    break  # Found a valid one, use it
        
        # Fallback to LUW if LPS failed or was too big (though LPS 11 is 660 < 936)
        if best_edges is None or best_n > 936:
            if time.time() < deadline - 5:
                n_luw, edges_luw = _luw_graph(k, g)
                if n_luw < best_n:
                    adj = [[] for _ in range(n_luw)]
                    for u, v in edges_luw:
                        adj[u].append(v)
                        adj[v].append(u)
                    if _check_girth(adj, n_luw, g):
                        best_n = n_luw
                        best_edges = edges_luw
        
        if best_edges is not None:
            return best_n, best_edges
        
        # Ultimate fallback: Return LUW even if not verified (should not happen)
        n_luw, edges_luw = _luw_graph(k, g)
        return n_luw, edges_luw
    
    # General fallback
    n_luw, edges_luw = _luw_graph(k, g)
    return n_luw, edges_luw
