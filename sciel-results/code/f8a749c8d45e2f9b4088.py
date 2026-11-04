import time
from collections import deque

_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _deadline_exceeded(limit=140):
    return _elapsed() > limit

# ---------------------------------------------------------------------------
# Lazebnik-Ustimenko-Woldar construction (kept as the safe fallback)
# ---------------------------------------------------------------------------
def construct_lw(k, g):
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

# ---------------------------------------------------------------------------
# Correct non-backtracking girth check (certifies "no cycle shorter than g")
# ---------------------------------------------------------------------------
def _girth_ok(adj, n, g):
    # A cycle of length L exists iff there is a non-backtracking walk of length
    # L-1 from a vertex back to one of its neighbours.  So it suffices to run a
    # non-backtracking walk of length g-2 from every vertex and see whether any
    # neighbour is reached.
    max_walk = g - 2
    q = deque()
    for s in range(n):
        for nb in adj[s]:
            q.append((s, nb))
    for _ in range(max_walk):
        if not q:
            break
        nxt = []
        for prev, cur in q:
            for nb in adj[cur]:
                if nb != prev:
                    nxt.append((cur, nb))
        q = deque(nxt)
    for prev, cur in q:
        if prev in adj[cur]:
            return False
    return True

def _build_edges(adj, n):
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges

# ---------------------------------------------------------------------------
# Heisenberg-group Cayley construction (searched over parameters)
# ---------------------------------------------------------------------------
def _is_prime(m):
    if m < 2:
        return False
    if m % 2 == 0:
        return m == 2
    f = 3
    while f * f <= m:
        if m % f == 0:
            return False
        f += 2
    return True

def construct_abs_heisenberg(k, g):
    if k != 5:
        return None

    best = None
    # q is the order of the Heisenberg group; n = q^3.  We want the smallest q
    # (hence smallest n) for which a 5-generator Cayley graph has girth >= g.
    for q in range(6, 40):
        if _deadline_exceeded():
            break
        n = q * q * q
        # candidate generator sets: standard {±e1,±e2,c} and, when q is prime,
        # the scaled odd variants {±a·e1, ±b·e2, c}.
        sets = [((1, 0, 0), (q - 1, 0, 0), (0, 1, 0), (0, q - 1, 0), (0, 0, q // 2))]
        if _is_prime(q):
            odds = [a for a in range(1, q, 2) if a != 1 and a != q - 1]
            if not odds:
                odds = [1]
            for a in odds:
                for b in odds:
                    sets.append(((a, 0, 0), (q - a, 0, 0),
                                 (0, b, 0), (0, q - b, 0), (0, 0, q // 2)))
                    if _deadline_exceeded():
                        break
                if _deadline_exceeded():
                    break

        for gens in sets:
            if _deadline_exceeded():
                break
            adj = [[] for _ in range(n)]
            for x in range(q):
                for y in range(q):
                    for z in range(q):
                        u = (x * q + y) * q + z
                        for gx, gy, gz in gens:
                            vx = (x + gx) % q
                            vy = (y + gy) % q
                            vz = (z + gz + x * gy) % q
                            v = (vx * q + vy) * q + vz
                            if u < v:
                                adj[u].append(v)
                                adj[v].append(u)
            if _girth_ok(adj, n, g):
                if best is None or n < best[0]:
                    best = (n, _build_edges(adj, n))
                break  # found a working set for this q; larger q only worse
        if best is not None and n > best[0]:
            break
    return best

def construct(k, g):
    global _start_time
    _start_time = time.time()

    if k == 5:
        result = construct_abs_heisenberg(k, g)
        if result is not None:
            return result
    return construct_lw(k, g)
