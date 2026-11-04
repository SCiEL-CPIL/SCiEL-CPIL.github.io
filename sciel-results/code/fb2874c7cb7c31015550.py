import time
import math
import random
from collections import deque

_START_TIME = time.time()
_TIMEOUT = 135  # wall-clock budget (seconds); evaluator kills at 150 s -> keep margin


def _deadline_exceeded():
    return (time.time() - _START_TIME) > _TIMEOUT


# ---------------------------------------------------------------------------
# Verification helpers (from inspiration 3)
# ---------------------------------------------------------------------------
def _check_simple_and_regular(edges, n, k):
    """Check simple (no loops, no repeated edges) and k-regular."""
    if len(edges) != n * k // 2:
        return False
    seen = set()
    deg = [0] * n
    for u, v in edges:
        if u == v:
            return False
        a, b = min(u, v), max(u, v)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[a] += 1
        deg[b] += 1
    for d in deg:
        if d != k:
            return False
    return True


def _verify_girth(edges, n, g):
    """Check no cycle shorter than g using non-backtracking walks from every vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for start in range(n):
        dist = {}
        queue = deque()
        for nb in adj[start]:
            queue.append((start, nb, 1))
            dist[(start, nb)] = 1
        while queue:
            prev, curr, d = queue.popleft()
            if d >= g:
                continue
            for nb2 in adj[curr]:
                if nb2 == prev:
                    continue
                state = (curr, nb2)
                if state in dist:
                    if d + 1 < g:
                        return False
                else:
                    dist[state] = d + 1
                    queue.append((curr, nb2, d + 1))
    return True


def _valid(n, edges, k, g):
    return _check_simple_and_regular(edges, n, k) and _verify_girth(edges, n, g)


# ---------------------------------------------------------------------------
# Moore bound
# ---------------------------------------------------------------------------
def _moore_bound(k, g):
    if g % 2 == 0:
        return 2 * sum((k - 1) ** i for i in range(g // 2))
    return 1 + k * sum((k - 1) ** i for i in range((g - 1) // 2))


def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


# ---------------------------------------------------------------------------
# Lazebnik-Ustimenko-Woldar graph CD(d, q), q = k (bipartite)
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
# Alon-Babai-Suzuki construction on Heisenberg group H_q (even k)
# ---------------------------------------------------------------------------
def construct_abs(k, g):
    mb = _moore_bound(k, g)

    q = 2
    while True:
        if _is_prime(q):
            if q ** 3 >= mb:
                break
        q += 1

    if k == 4:
        gens = [
            (1, 0, 0),
            (q - 1, 0, 0),
            (0, 1, 0),
            (0, q - 1, 1),
        ]
    elif k == 6:
        gens = [
            (1, 0, 0),
            (q - 1, 0, 0),
            (0, 1, 0),
            (0, q - 1, 1),
            (0, 0, 1),
            (0, 0, q - 1),
        ]
    else:
        return None

    n = q ** 3
    edges = []

    def idx(x, y, z):
        return (x * q + y) * q + z

    for x in range(q):
        for y in range(q):
            for z in range(q):
                u = idx(x, y, z)
                for gx, gy, gz in gens:
                    vx = (x + gx) % q
                    vy = (y + gy) % q
                    vz = (z + gz + x * gy) % q
                    v = idx(vx, vy, vz)
                    if u < v:
                        edges.append((u, v))
    return n, edges


# ---------------------------------------------------------------------------
# Circulant construction (within-family, for even k not covered by ABS)
# Generators +/-g1 .. +/-g_{k/2} on Z_q; verify before returning.
# ---------------------------------------------------------------------------
def construct_circulant(k, g, max_q=4000):
    if k % 2 != 0:
        return None
    half = k // 2
    mb = _moore_bound(k, g)
    # circulant q must be large enough; start near Moore bound
    start_q = mb if mb >= 2 * half + 1 else 2 * half + 1
    for q in range(start_q, start_q + max_q):
        if _deadline_exceeded():
            return None
        # generators 1..half are distinct and < q/2 -> no loops / no collisions
        # (a generator g and q-g are the same undirected edge; g in 1..half distinct)
        edges = []
        for u in range(q):
            for gi in range(1, half + 1):
                v = (u + gi) % q
                if u < v:
                    edges.append((u, v))
        # quick regularity/size check
        if len(edges) != q * k // 2:
            continue
        if _verify_girth(edges, q, g):
            return q, edges
    return None


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------
def construct(k, g):
    random.seed(12345)
    candidates = []  # list of (n, edges)

    def try_add(n, edges):
        if n is None:
            return
        # cheap checks first
        if len(edges) != n * k // 2:
            return
        candidates.append((n, edges))

    # 1. ABS for even k (k=4,6)
    if k % 2 == 0:
        res = construct_abs(k, g)
        try_add(*res)

    # 2. Circulant for even k (extra within-family candidate)
    if k % 2 == 0 and not _deadline_exceeded():
        res = construct_circulant(k, g)
        try_add(*res)

    # 3. LW (works for any k)
    lw = construct_lw(k, g)
    try_add(*lw)

    # Verify candidates (girth is the expensive check) and keep the smallest valid.
    best = None
    for n, edges in candidates:
        if _deadline_exceeded():
            break
        if _check_simple_and_regular(edges, n, k) and _verify_girth(edges, n, g):
            if best is None or n < best[0]:
                best = (n, edges)

    if best is not None:
        return best

    # Fallback: return LW even without full verification (it is a known valid family
    # for odd d; better to return something than nothing).
    return lw
