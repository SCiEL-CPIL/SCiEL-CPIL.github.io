import time
from collections import deque

def _luw_construct(q, d):
    """Build LUW graph CD(d, q). Returns (n, edges)."""
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_fn = lambda x, y: x ^ y
        sub_fn = lambda x, y: x ^ y
        mul_fn = lambda x, y: table[x][y]
    else:
        add_fn = lambda x, y: (x + y) % q
        sub_fn = lambda x, y: (x - y) % q
        mul_fn = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
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


def _luw_construct_with_d(q, d):
    """Alias for _luw_construct."""
    return _luw_construct(q, d)


def _luw_construct_original(k, g):
    """Original LUW construction as fallback."""
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _luw_construct(k, d)


def _check_girth_fast(n, edges, g):
    """
    Check if graph with n vertices and given edges has girth >= g.
    Uses BFS from each vertex. Stops early if a cycle < g is found.
    """
    if n < g:
        return False

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    max_depth = (g - 1) // 2

    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])

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
                    if cycle_len < g:
                        return False
    return True


def _check_degree(n, edges, k):
    """Verify all vertices have degree exactly k."""
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return all(d == k for d in deg)


def _lex_product(n1, edges1, n2, edges2, k):
    """
    Lexicographic product of two k-regular graphs.
    G1 x G2: vertices (u, v) for u in G1, v in G2.
    (u1, v1) ~ (u2, v2) iff u1 ~ u2 (any v1, v2) OR u1 == u2 and v1 ~ v2.
    Result is k-regular.
    """
    n = n1 * n2
    edges = []

    # Edges from G2 (within same G1 vertex)
    for u in range(n1):
        for v1, v2 in edges2:
            edges.append((u * n2 + v1, u * n2 + v2))

    # Edges from G1 (between different G1 vertices, all G2 pairs)
    for u1, u2 in edges1:
        for v1 in range(n2):
            for v2 in range(n2):
                edges.append((u1 * n2 + v1, u2 * n2 + v2))

    return n, edges


def _check_girth_lex(n1, edges1, n2, edges2, g, time_limit):
    """
    Check girth of lexicographic product without building it explicitly.
    A cycle in G1 x G2 projects to either:
    1. A cycle in G2 (length = cycle in G2)
    2. A walk in G1 with lifts in G2
    """
    start = time.time()

    # First check: girth of G2
    if not _check_girth_fast(n2, edges2, g):
        return False

    # Check girth of G1 (cycles in G1 lift to cycles in product)
    if not _check_girth_fast(n1, edges1, g):
        return False

    # The tricky part: cycles that project to walks in G1
    # A cycle of length L in the product projects to a closed walk of length L in G1
    # where consecutive edges in G1 may be "backtracking" (going to same neighbor)
    # and the G2 components must connect properly.
    #
    # For a bipartite k-regular graph, the shortest cycle in the lexicographic
    # product is at least min(girth(G1), girth(G2), 2*k) if both are bipartite
    # and large enough. Actually it's more subtle.
    #
    # Let's just do a bounded BFS on the product graph up to depth (g-1)//2
    # But we do it smartly: we only explore up to depth max_depth.

    max_depth = (g - 1) // 2
    if max_depth < 1:
        return True

    # Build adjacency for G1 and G2
    adj1 = [[] for _ in range(n1)]
    for u, v in edges1:
        adj1[u].append(v)
        adj1[v].append(u)

    adj2 = [[] for _ in range(n2)]
    for u, v in edges2:
        adj2[u].append(v)
        adj2[v].append(u)

    # BFS from (0, 0) in the product graph, but we can't store all states.
    # Instead, we do a non-backtracking walk check up to depth max_depth.
    # For a 3-regular graph, the number of non-backtracking walks of length l
    # from a vertex is 3 * 2^(l-1). For l=7, that's 3*64=192 per starting vertex.
    # Total: n * 192 which for n up to ~10000 is ~2M. Manageable.

    # Actually, let's just check if there's any cycle < g by doing BFS
    # from each vertex up to depth max_depth. But we need to be careful
    # about the product structure.

    # Simpler approach: just build the product and check if it's small enough
    if n1 * n2 <= 50000:
        n, edges = _lex_product(n1, edges1, n2, edges2, k)
        return _check_girth_fast(n, edges, g)
    else:
        # Too large to build explicitly, use theoretical bound
        # For lexicographic product of two bipartite graphs with girths g1, g2:
        # The girth is at least min(g1, g2) but can be smaller due to "mixed" cycles.
        # A mixed cycle of length 2m: goes u1->u2 in G1, v1->v2 in G2, back...
        # Actually the shortest mixed cycle has length at least 4 if both graphs
        # have edges. For k-regular bipartite, the girth of the product is at
        # least 4 (a 4-cycle: (u1,v1)-(u2,v1)-(u2,v2)-(u1,v2)-(u1,v1)).
        # Wait, that's a 4-cycle! So the lexicographic product always has 4-cycles
        # if both graphs have at least one edge.
        #
        # Hmm, that means lexicographic product doesn't help for girth >= 16.
        # Let me reconsider.
        return False


def _cartesian_product(n1, edges1, n2, edges2, k):
    """
    Cartesian product of two graphs.
    (u1, v1) ~ (u2, v2) iff (u1==u2 and v1~v2) OR (v1==v2 and u1~u2).
    If both are k-regular, result is 2k-regular. Not what we want.
    """
    # This gives 2k-regular, not k-regular. Skip.
    pass


def _tensor_product(n1, edges1, n2, edges2, k):
    """
    Tensor (categorical/direct) product.
    (u1, v1) ~ (u2, v2) iff u1~u2 AND v1~v2.
    If both are k-regular, result is k-regular.
    """
    n = n1 * n2
    edges = []
    for u1, u2 in edges1:
        for v1, v2 in edges2:
            edges.append((u1 * n2 + v1, u2 * n2 + v2))
    return n, edges


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    if k != 3 or g != 16:
        return _luw_construct_original(k, g)

    q = k
    best_result = None

    # Strategy 1: Sweep d in LUW construction to find smallest d with girth >= g
    # LUW CD(d, 3) has ~2*3^d vertices
    # d=5: ~486, d=6: ~1458, d=7: ~4374, d=8: ~13122
    for d in range(5, 10):
        if time.time() > deadline - 10:
            break

        try:
            n, edges = _luw_construct(q, d)
        except Exception:
            continue

        if n < 2:
            continue

        if not _check_degree(n, edges, k):
            continue

        if _check_girth_fast(n, edges, g):
            best_result = (n, edges)
            break

    # Strategy 2: Try tensor product of two smaller LUW graphs
    # Tensor product of two 3-regular graphs is 3-regular.
    # If G1 has girth g1 and G2 has girth g2, the tensor product
    # has girth at least min(g1, g2) but could be smaller.
    # Actually for bipartite graphs, tensor product girth = min(g1, g2)
    # if both are large enough. Let's try.
    if time.time() < deadline - 20:
        for d1 in range(4, 7):
            for d2 in range(4, 7):
                if time.time() > deadline - 15:
                    break
                if d1 + d2 < 8:
                    continue

                try:
                    n1, edges1 = _luw_construct(q, d1)
                    n2, edges2 = _luw_construct(q, d2)
                except Exception:
                    continue

                n = n1 * n2
                if n > 50000:
                    continue

                # Check if both have girth >= g (necessary but not sufficient)
                if not _check_girth_fast(n1, edges1, g):
                    continue
                if not _check_girth_fast(n2, edges2, g):
                    continue

                try:
                    n_t, edges_t = _tensor_product(n1, edges1, n2, edges2, k)
                except Exception:
                    continue

                if not _check_degree(n_t, edges_t, k):
                    continue

                if _check_girth_fast(n_t, edges_t, g):
                    if best_result is None or n_t < best_result[0]:
                        best_result = (n_t, edges_t)

    if best_result is not None:
        return best_result

    # Fallback: use larger d in LUW
    for d in range(10, 14):
        if time.time() > deadline - 5:
            break
        try:
            n, edges = _luw_construct(q, d)
            if _check_degree(n, edges, k) and _check_girth_fast(n, edges, g):
                return n, edges
        except Exception:
            continue

    # Final fallback
    return _luw_construct_original(k, g)


def _check_girth(n, edges, g):
    """Alias for _check_girth_fast."""
    return _check_girth_fast(n, edges, g)
