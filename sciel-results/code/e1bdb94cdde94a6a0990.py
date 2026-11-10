import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses SL(2, q) Cayley graph approach with optimized girth checking.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    if k != 3:
        # Fallback: try LUW construction for other k values
        return _luw_construct(k, g, start_time, deadline)

    # 1. Define GF(16) arithmetic
    m = 4
    q = 1 << m
    poly = 0b10011  # x^4 + x + 1

    # Precompute multiplication table for GF(16)
    mul_table = [[0] * q for _ in range(q)]
    for i in range(q):
        for j in range(q):
            res = 0
            a = i
            b = j
            while b:
                if b & 1:
                    res ^= a
                a <<= 1
                if a >= q:
                    a ^= poly
                b >>= 1
            mul_table[i][j] = res

    # Precompute inverse table
    inv_table = [0] * q
    for i in range(1, q):
        for j in range(1, q):
            if mul_table[i][j] == 1:
                inv_table[i] = j
                break

    def gf_mul(a, b):
        return mul_table[a][b]

    def gf_add(a, b):
        return a ^ b

    def gf_inv(a):
        return inv_table[a] if a != 0 else 0

    def gf_div(a, b):
        if b == 0:
            return 0
        return gf_mul(a, gf_inv(b))

    # 2. Generate SL(2, 16) elements
    elements = []
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0:
                    if gf_mul(b, c) == 1:
                        for d in range(q):
                            elements.append((a, b, c, d))
                else:
                    bc = gf_mul(b, c)
                    rhs = gf_add(1, bc)
                    d = gf_div(rhs, a)
                    elements.append((a, b, c, d))

    n = len(elements)
    elem_to_idx = {e: i for i, e in enumerate(elements)}

    # 3. Find Involutions
    involutions = []
    for e in elements:
        a, b, c, d = e
        if gf_add(a, d) == 0 and not (a == 1 and b == 0 and c == 0 and d == 1):
            involutions.append(e)

    def multiply(m1, m2):
        a1, b1, c1, d1 = m1
        a2, b2, c2, d2 = m2
        aa = gf_add(gf_mul(a1, a2), gf_mul(b1, c2))
        ab = gf_add(gf_mul(a1, b2), gf_mul(b1, d2))
        ba = gf_add(gf_mul(c1, a2), gf_mul(d1, c2))
        bb = gf_add(gf_mul(c1, b2), gf_mul(d1, d2))
        return (aa, ab, ba, bb)

    def check_girth_fast(adj, n_nodes, target_girth):
        """
        Optimized non-backtracking girth check using BFS with parent tracking.
        """
        max_depth = (target_girth - 1) // 2
        for s in range(n_nodes):
            dist = [-1] * n_nodes
            parent = [-1] * n_nodes
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = du + dist[v] + 1
                        if cycle_len < target_girth:
                            return False
        return True

    # Use a fixed seed for reproducibility
    random.seed(42)

    # Pre-compute all pairwise products for faster adjacency construction
    max_trials = 50
    best_n = n
    best_edges = None

    for trial in range(max_trials):
        if time.time() > deadline - 30:
            break

        S = random.sample(involutions, 3)

        # Build adjacency
        adj = [[] for _ in range(n)]
        for i, e in enumerate(elements):
            for s in S:
                f = multiply(e, s)
                j = elem_to_idx[f]
                adj[i].append(j)

        # Quick connectivity check
        visited = [False] * n
        queue = deque([0])
        visited[0] = True
        count = 1
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    queue.append(v)

        if count != n:
            continue

        # Check girth
        if check_girth_fast(adj, n, g):
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges

    # Fallback: LUW construction
    return _luw_construct(k, g, start_time, deadline)

def _luw_construct(k, g, start_time, deadline):
    """LUW construction as fallback."""
    def build_luw(d, q_val):
        if q_val == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add_fn = lambda x, y: x ^ y
            sub_fn = lambda x, y: x ^ y
            mul_fn = lambda x, y: table[x][y]
        else:
            add_fn = lambda x, y: (x + y) % q_val
            sub_fn = lambda x, y: (x - y) % q_val
            mul_fn = lambda x, y: x * y % q_val

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                    else:
                        y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def check_girth_luw(n, edges, girth_target):
        max_depth = (girth_target - 1) // 2
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < girth_target:
                            return False
        return True

    q_val = 3
    best_luw = None
    for d in range(5, 15):
        if time.time() > deadline:
            break
        n_luw, edges_luw = build_luw(d, q_val)
        if n_luw < 2:
            continue
        deg = [0] * n_luw
        for u, v in edges_luw:
            deg[u] += 1
            deg[v] += 1
        if any(dd != k for dd in deg):
            continue
        if check_girth_luw(n_luw, edges_luw, g):
            return n_luw, edges_luw
        best_luw = (n_luw, edges_luw)

    if best_luw:
        return best_luw

    return 0, []
# EVOLVE-BLOCK-END
