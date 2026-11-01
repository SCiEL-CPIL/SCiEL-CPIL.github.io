import time
import math
from collections import deque

def construct(k, g):
    start = time.time()
    deadline = start + 135.0

    def time_left():
        return deadline - time.time()

    def build_luw(q, d_val):
        # Lazebnik-Ustimenko-Woldar construction
        # Vertices: (side, x_1, ..., x_d) where side in {0,1}, x_i in {0,...,q-1}
        # Total vertices: 2 * q^d
        
        if q == 4:
            gf4_mul = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            def gf4_add(a, b): return a ^ b
            def gf4_sub(a, b): return a ^ b
            def gf4_mul_fn(a, b): return gf4_mul[a][b]
            field_add = gf4_add
            field_sub = gf4_sub
            field_mul = gf4_mul_fn
        else:
            def field_add(a, b): return (a + b) % q
            def field_sub(a, b): return (a - b) % q
            def field_mul(a, b): return (a * b) % q

        # Rule for LUW: for each c in 2..d, rule[c] = (a, b) meaning
        # side 0: y[c] = x[c] + y[a] * x[b]
        # side 1: y[c] = x[c] - x[a] * y[b]
        # Standard LUW rule:
        if d_val == 2:
            rule = {2: (1, 1)}
        elif d_val == 3:
            rule = {2: (1, 1), 3: (2, 1)}
        elif d_val == 4:
            rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        else:
            rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
            for c in range(5, d_val + 1):
                # Use a systematic rule
                rule[c] = (c - 1, 1)

        def get_neighbors(side, x):
            """Get neighbors of vertex (side, x) where x is tuple of length d"""
            out = []
            for first in range(q):
                y = list(x)
                y[0] = first  # y[1] in 1-indexed = y[0] in 0-indexed
                if side == 0:
                    for c in range(1, d_val):  # c is 0-indexed, corresponds to c+1 in 1-indexed
                        a, b = rule[c + 1]  # 1-indexed rule
                        a_idx = a - 1  # convert to 0-indexed
                        b_idx = b - 1
                        y[c] = field_add(x[c], field_mul(y[a_idx], x[b_idx]))
                else:
                    for c in range(1, d_val):
                        a, b = rule[c + 1]
                        a_idx = a - 1
                        b_idx = b - 1
                        y[c] = field_sub(x[c], field_mul(x[a_idx], y[b_idx]))
                out.append((1 - side, tuple(y)))
            return out

        # Enumerate all vertices
        all_vertices = []
        index = {}
        for side in range(2):
            for x_tuple in _product(range(q), d_val):
                v = (side, x_tuple)
                index[v] = len(all_vertices)
                all_vertices.append(v)

        n = len(all_vertices)
        edges = []
        for v in all_vertices:
            vid = index[v]
            side, x = v
            for u in get_neighbors(side, x):
                uid = index[u]
                if vid < uid:
                    edges.append((vid, uid))

        return n, edges

    def _product(iterable, repeat):
        """Generate cartesian product (like itertools.product)"""
        pools = [list(iterable)] * repeat
        result = [[]]
        for pool in pools:
            result = [x + [y] for x in result for y in pool]
        return [tuple(x) for x in result]

    def verify_fast(n, edges, k, g):
        """Fast verification using BFS from each vertex up to depth g//2"""
        adj = [[] for _ in range(n)]
        seen = set()
        for u, v in edges:
            if u == v or u < 0 or u >= n or v < 0 or v >= n:
                return False
            key = (min(u,v), max(u,v))
            if key in seen:
                return False
            seen.add(key)
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        max_depth = g // 2
        for u in range(n):
            if time.time() > deadline - 10.0:
                return False
            dist = [-1] * n
            dist[u] = 0
            q_ = deque([u])
            while q_:
                curr = q_.popleft()
                if dist[curr] >= max_depth:
                    continue
                for v in adj[curr]:
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        q_.append(v)
                    else:
                        if dist[v] >= dist[curr]:
                            cycle_len = dist[curr] + 1 + dist[v]
                            if cycle_len < g:
                                return False
        return True

    best_n = float('inf')
    best_edges = None

    # For k=4, g=10: Moore bound is 4*(1+4+16+64+256) = 1364
    # LUW with q=4, d=5 gives n=2*4^5=2048
    # LUW with q=5, d=4 gives n=2*5^4=1250 (below Moore bound, so girth < 10)
    # Try various (q, d) combinations
    
    candidates = []
    if k == 4 and g == 10:
        # Known best is 320, but let's try LUW first
        candidates.append((4, 5))  # n=2048
        candidates.append((5, 4))  # n=1250
        candidates.append((3, 6))  # n=2*729=1458
        candidates.append((4, 4))  # n=512 (girth may be < 10)
    else:
        min_d = max(3, (g - 2) // 2)
        for q in [3, 4, 5, 7]:
            for d_val in range(min_d, min_d + 5):
                candidates.append((q, d_val))

    for q, d_val in candidates:
        if time_left() < 30:
            break
        try:
            n, edges = build_luw(q, d_val)
            if n < best_n and n <= 5000:
                if verify_fast(n, edges, k, g):
                    best_n = n
                    best_edges = edges
        except Exception:
            pass

    if best_edges is not None:
        return best_n, best_edges

    return 0, []
