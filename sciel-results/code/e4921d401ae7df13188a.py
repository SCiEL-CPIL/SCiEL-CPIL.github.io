import time
import math

def construct(k, g):
    deadline = time.time() + 115
    if k == 4 and g == 10:
        result = _construct_4_10_cage(deadline)
        if result is not None:
            return result
    luw_result = _luw_construct(k, g)
    if luw_result is not None:
        n_luw, edges_luw = luw_result
        if _verify_graph(edges_luw, n_luw, k, g):
            return n_luw, edges_luw
    return luw_result

def _construct_4_10_cage(deadline=None):
    N = 320
    if deadline is None:
        deadline = time.time() + 110
    candidates_a = [1]
    for a in range(2, N // 4 + 1):
        if math.gcd(a, N) == 1:
            candidates_a.append(a)
        if time.time() > deadline - 30:
            break
    for a in candidates_a:
        if time.time() > deadline - 10:
            break
        for b in range(a + 1, N // 2 + 1):
            if time.time() > deadline - 5:
                break
            if math.gcd(b, N) != 1:
                continue
            if (b + a) % N == 0:
                continue
            if _has_short_cycle(N, a, b, 4) or _has_short_cycle(N, a, b, 6) or _has_short_cycle(N, a, b, 8):
                continue
            edges = _build_cayley_edges(N, a, b)
            if _verify_graph_fast(edges, N, 4, 10):
                return N, edges
    return None

def _build_cayley_edges(N, a, b):
    edges = []
    edge_set = set()
    for i in range(N):
        for offset in (a, -a, b, -b):
            j = (i + offset) % N
            if i < j:
                key = (i, j)
                if key not in edge_set:
                    edge_set.add(key)
                    edges.append(key)
    return edges

def _has_short_cycle(N, a, b, L):
    for u in range(-L, L + 1):
        for v in range(-L, L + 1):
            if (u * a + v * b) % N != 0:
                continue
            min_A = abs(u)
            max_A = L - abs(v)
            if min_A > max_A:
                continue
            A = min_A
            if A % 2 != u % 2:
                A += 1
            if A > max_A:
                continue
            B = L - A
            if B >= abs(v) and B % 2 == v % 2:
                return True
    return False

def _luw_construct(k, g):
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

def _verify_graph_fast(edges, n, k, g):
    if n <= 0 or len(edges) == 0:
        return False
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        if len(adj[i]) != k or len(set(adj[i])) != k:
            return False
    max_depth = (g - 2) // 2
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        for d in range(1, max_depth + 1):
            new_head = len(queue)
            while head < new_head:
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = d
                        parent[v] = u
                        queue.append(v)
                    else:
                        if dist[v] < d - 1:
                            if d + dist[v] + 1 < g:
                                return False
                        elif dist[v] == d - 1 and parent[u] != v:
                            if 2 * d < g:
                                return False
    return True

def _verify_graph(edges, n, k, g):
    return _verify_graph_fast(edges, n, k, g)

def _random_search(k, g, deadline):
    return _luw_construct(k, g)
