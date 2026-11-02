import time
import random
from collections import deque

_DEADLINE = 115.0  # safety margin vs 120s wall-clock


def _girth_bfs(n, adj, limit):
    """Exact girth via non-backtracking BFS from every vertex. Returns True if
    girth >= limit."""
    for s in range(n):
        dist = {s: 0}
        prev = {s: -1}
        dq = deque([s])
        while dq:
            u = dq.popleft()
            du = dist[u]
            if du + 1 >= limit:
                continue
            for v in adj[u]:
                if v == prev.get(u, -1):
                    continue
                if v not in dist:
                    dist[v] = du + 1
                    prev[v] = u
                    dq.append(v)
                else:
                    if du + 1 + dist[v] < limit:
                        return False
    return True


def _bipartite_luw(q, d):
    """Incidence graph of Lazebnik-Ustimenko-Woldar CD(d,q)."""
    q = int(q)
    d = int(d)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
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


def _has_short_cycle_3(n, a, b, c):
    """Check for cycles of length 3 in Cayley graph Z_n with generators {a, -a, b, -b, c}."""
    # a + b + c = 0 => a + b = n/2
    if (a + b) % n == c or (a - b) % n == c or (-a + b) % n == c or (-a - b) % n == c:
        return True
    # 2a + b = 0 => 2a = -b
    if (2 * a + b) % n == 0 or (2 * a - b) % n == 0:
        return True
    # a + 2b = 0 => a = -2b
    if (a + 2 * b) % n == 0 or (a - 2 * b) % n == 0:
        return True
    # 2a + c = 0 => 2a = -c = n/2
    if (2 * a + c) % n == 0 or (2 * a - c) % n == 0:
        return True
    # 2b + c = 0 => 2b = -c = n/2
    if (2 * b + c) % n == 0 or (2 * b - c) % n == 0:
        return True
    # 3a = 0
    if (3 * a) % n == 0:
        return True
    # 3b = 0
    if (3 * b) % n == 0:
        return True
    # 3c = 0 (c = n/2, 3n/2 != 0 mod n for even n)
    return False

def _has_short_cycle_4(n, a, b, c):
    """Check for non-backtracking cycles of length 4."""
    # a + a - a - a = 0 (backtracking, ignored)
    # a + b - a - b = 0 (backtracking, ignored)
    # a + a + b - b = 0 => 2a = 0
    if (2 * a) % n == 0:
        return True
    # b + b + a - a = 0 => 2b = 0
    if (2 * b) % n == 0:
        return True
    # a + b + c - c = 0 => a + b = 0 => a = -b
    if (a + b) % n == 0:
        return True
    # a + c + b - c = 0 => a + b = 0
    # a + c + c - a = 0 => 2c = 0 => n = 0
    # a + a + c - c = 0 => 2a = 0
    # a + b + a - b = 0 => 2a = 0
    # a + b + b - a = 0 => 2b = 0
    # a + c + b - b = 0 => a + c = 0 => a = -c = n/2
    if (a + c) % n == 0:
        return True
    # b + c + a - a = 0 => b + c = 0 => b = -c = n/2
    if (b + c) % n == 0:
        return True
    return False

def _search_cayley(k, g, max_n, deadline):
    """Search for a small k-regular Cayley graph on Z_n with girth >= g."""
    if k != 5:
        return None
    
    rng = random.Random(123)
    # Moore bound for k=5, g=9 is 426
    # We search n in [426, max_n)
    # n must be even for 5-regular graph on Z_n with symmetric generators?
    # 5n must be even => n even.
    
    for n in range(426, max_n, 2):
        if time.time() > deadline - 5:
            break
        c = n // 2
        # Try more random pairs
        for _ in range(100):
            a = rng.randint(1, n - 1)
            b = rng.randint(1, n - 1)
            if a == b or a == n - b or b == n - a or a == c or b == c:
                continue
            if _has_short_cycle_3(n, a, b, c):
                continue
            if _has_short_cycle_4(n, a, b, c):
                continue
            
            adj = [[] for _ in range(n)]
            for u in range(n):
                adj[u].extend([(u + a) % n, (u - a) % n, (u + b) % n, (u - b) % n, (u + c) % n])
            
            if _girth_bfs(n, adj, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
    return None


def construct(k, g):
    t0 = time.time()
    deadline = t0 + _DEADLINE

    # Baseline: LUW construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n_luw, edges_luw = _bipartite_luw(q, d)
    
    best_n, best_edges = n_luw, edges_luw
    
    # Try to find a smaller Cayley graph
    if k == 5 and g == 9:
        result = _search_cayley(k, g, best_n, deadline)
        if result:
            n_cay, edges_cay = result
            if n_cay < best_n:
                best_n, best_edges = n_cay, edges_cay
                
    return best_n, best_edges
