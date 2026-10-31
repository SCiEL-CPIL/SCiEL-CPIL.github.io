import time
from collections import deque

# ---- global deadline (150s wall-clock kill; keep margin) ----
_T0 = time.time()
_DEADLINE = _T0 + 135.0


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1  # make it odd
    return _luw_construct_with_d(k, g, d)


def _luw_construct_with_d(k, g, d):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with specified d."""
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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


def _verify_girth(adj, g):
    """Check that graph with adjacency list 'adj' has girth >= g.
    BFS from each vertex to depth (g-1)//2, tracking parent to detect cycles."""
    n = len(adj)
    max_depth = (g - 1) // 2
    for start in range(n):
        visited = {}
        visited[start] = (0, -1)
        queue = deque([(start, 0, -1)])
        while queue:
            node, depth, parent = queue.popleft()
            if depth >= max_depth:
                continue
            for nb in adj[node]:
                if nb == parent:
                    continue
                if nb in visited:
                    cycle_len = depth + 1 + visited[nb][0]
                    if cycle_len < g:
                        return False
                else:
                    visited[nb] = (depth + 1, node)
                    queue.append((nb, depth + 1, node))
    return True


def _verify_girth_edges(n, edges, g):
    """Build adjacency list from edges and verify girth >= g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return _verify_girth(adj, g)


def _has_short_cycle_circulant(n, a, b, g):
    """Exact girth check for C_n(1,a,b) using a single BFS from vertex 0
    (valid because the graph is vertex-transitive).
    Returns True if a cycle shorter than g exists."""
    visited = [-1] * n
    parent = [-1] * n
    visited[0] = 0
    queue = deque([(0, 0)])
    max_depth = g // 2
    shifts = (1, a, b)
    while queue:
        u, depth = queue.popleft()
        for shift in shifts:
            for v in ((u + shift) % n, (u - shift) % n):
                if visited[v] == -1:
                    visited[v] = depth + 1
                    parent[v] = u
                    if depth + 1 <= max_depth:
                        queue.append((v, depth + 1))
                else:
                    if parent[v] == u or parent[u] == v:
                        continue
                    if depth + 1 + visited[v] < g:
                        return True
    return False


def _quick_4cycle_check(n, a, b):
    """Necessary condition for a 4-cycle in C_n(1,a,b)."""
    gens = (1, -1, a, -a, b, -b)
    for g1 in gens:
        for g2 in gens:
            s = (g1 + g2) % n
            if s != 0 and s in gens:
                return True
    return False


def _quick_6cycle_check(n, a, b):
    """Necessary condition for a 6-cycle in C_n(1,a,b)."""
    gens = (1, -1, a, -a, b, -b)
    for g1 in gens:
        for g2 in gens:
            for g3 in gens:
                s = (g1 + g2 + g3) % n
                if s != 0 and s in gens:
                    return True
    return False


def _quick_8cycle_check(n, a, b):
    """Necessary condition for an 8-cycle in C_n(1,a,b)."""
    gens = (1, -1, a, -a, b, -b)
    for g1 in gens:
        for g2 in gens:
            for g3 in gens:
                for g4 in gens:
                    s = (g1 + g2 + g3 + g4) % n
                    if s != 0 and s in gens:
                        return True
    return False


def _quick_10cycle_check(n, a, b):
    """Necessary condition for a 10-cycle in C_n(1,a,b)."""
    gens = (1, -1, a, -a, b, -b)
    for g1 in gens:
        for g2 in gens:
            for g3 in gens:
                for g4 in gens:
                    for g5 in gens:
                        s = (g1 + g2 + g3 + g4 + g5) % n
                        if s != 0 and s in gens:
                            return True
    return False


def _find_circulant_cage(k, g, deadline):
    """Search for a circulant graph C_n(1,a,b) with girth at least g (k=3)."""
    if k != 3:
        return None
    # Moore bound for n(3,16): 2*sum_{i=0}^{7} 2^i = 510; n even -> start 512.
    min_n = 512
    max_n = 940
    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 5:
            return None
        for a in range(2, n // 2):
            if time.time() > deadline - 5:
                return None
            for b in range(a + 1, n // 2):
                if _quick_4cycle_check(n, a, b):
                    continue
                if _quick_6cycle_check(n, a, b):
                    continue
                if _quick_8cycle_check(n, a, b):
                    continue
                if _quick_10cycle_check(n, a, b):
                    continue
                if _has_short_cycle_circulant(n, a, b, g):
                    continue
                edges = []
                for i in range(n):
                    for shift in (1, a, b):
                        j = (i + shift) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges
    return None


def construct(k, g):
    """Construct a k-regular graph with girth >= g.
    Returns (n, edges)."""
    if k == 3 and g == 16:
        deadline = _DEADLINE
        # Structural search: circulant graphs C_n(1,a,b) (vertex-transitive ->
        # girth certified by a single BFS, so we can sweep many candidates).
        try:
            result = _find_circulant_cage(k, g, deadline)
            if result is not None:
                n, edges = result
                # Final exact certification before returning.
                if _verify_girth_edges(n, edges, g):
                    return n, edges
        except Exception:
            pass
        # Guaranteed-valid fallback: LUW construction (always k-regular,
        # girth g by design for q>=3, d odd >= g-5).
        try:
            return _luw_construct(k, g)
        except Exception:
            pass
    # General case
    return _luw_construct(k, g)
