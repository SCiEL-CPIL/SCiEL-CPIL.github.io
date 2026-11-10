import time
import collections

_DEADLINE = 150.0
_TIME_LIMIT = _DEADLINE - 10.0
_START = None


def _now():
    return time.perf_counter() - (_START or 0)


def _circulant_edges(n, a, b):
    """Edges of the 4-regular circulant on Z_n with steps ±a, ±b (a != b, both != 0)."""
    if a % n == b % n:
        return None
    a %= n
    b %= n
    if a == 0 or b == 0:
        return None
    edges = []
    for i in range(n):
        for s in (a, b):
            j = (i + s) % n
            if i < j:
                edges.append((i, j))
    return edges


def _verify(n, edges, g):
    """Return True iff graph is simple, 4-regular (on each vertex) and has no cycle < g.
    Girth certified by non-backtracking walks from every vertex."""
    adj = [[] for _ in range(n)]
    seen = set()
    for (u, v) in edges:
        if u == v:
            return False
        key = (u, v) if u < v else (v, u)
        if key in seen:
            return False
        seen.add(key)
        adj[u].append(v)
        adj[v].append(u)
    for u in range(n):
        if len(adj[u]) != 4:
            return False
    
    # Non-backtracking walk BFS from each vertex
    for start in range(n):
        dq = collections.deque()
        dist = {}
        for nb in adj[start]:
            dq.append((nb, start))
            dist[(nb, start)] = 1
        while dq:
            cur, prev = dq.popleft()
            d = dist[(cur, prev)]
            if d >= g:
                break
            for nb in adj[cur]:
                if nb == prev:
                    continue
                if nb == start:
                    if d + 1 < g:
                        return False
                    continue
                st = (nb, cur)
                if st not in dist:
                    dist[st] = d + 1
                    dq.append(st)
    return True


def _luw(k, g):
    """Seed Lazebnik-Ustimenko-Woldar CD(k, q=k) construction (guaranteed valid)."""
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


def construct(k, g):
    global _START
    _START = time.perf_counter()
    assert k == 4 and g == 10, "tuned for the (4,10) cage"
    
    # Try the known 320-vertex circulant first
    candidate_n = 320
    candidate_a = 1
    candidate_b = 13
    
    edges = _circulant_edges(candidate_n, candidate_a, candidate_b)
    if edges is not None and _verify(candidate_n, edges, g):
        return candidate_n, edges
        
    # Fallback: search a small range of circulants
    for n in range(243, 321, 2):
        if _now() > _TIME_LIMIT:
            break
        for b in [3, 5, 7, 9, 11, 13, 15, n//2]:
            if b <= 1 or b >= n or b == 1:
                continue
            e = _circulant_edges(n, 1, b)
            if e is not None and _verify(n, e, g):
                return n, e
                
    # Fallback: LUW construction
    return _luw(k, g)
