"""Direct constructive solver for the (k, g)-cage problem (here k=5, g=9).

Approach
--------
* A *guaranteed-valid* 5-regular graph of girth >= g is built with the
  Lazebnik-Ustimenko-Woldar construction CD(d, q) (q-regular bipartite graph
  of girth d + 5; with d + 5 >= g it has no cycle shorter than g).
* To *improve* on that, a family of explicit 5-regular Cayley graphs on the
  abelian group Z_q x Z_2 is scanned in order of increasing vertex count
  n = 2 q.  Each candidate is verified for the girth requirement with the
  same bounded non-backtracking walk search the evaluator uses.  The first
  (smallest) verified-valid candidate in the family beats the LUW baseline.
* A wall-clock deadline guard returns the best verified-valid result found so
  far, with a safety margin before the evaluation limit.
"""
import time
from collections import deque

_DEADLINE = None


def _time_left(margin=15.0):
    if _DEADLINE is None:
        return float("inf")
    return _DEADLINE - time.monotonic() - margin


# --------------------------------------------------------------------------- #
# girth verification (exact, non-backtracking walks from every vertex)        #
# --------------------------------------------------------------------------- #
def _shortest_cycle(adj, n, g):
    INF = 1 << 30
    # We only need to detect cycles of length < g. A non-backtracking walk
    # from s of length d that returns to s yields a cycle of length d, so we
    # must search to depth g-1. Deeper states can never close a < g cycle, so
    # we never enqueue depth >= g.
    for s in range(n):
        seen = set()
        dq = deque()
        for w in adj[s]:
            seen.add((w, s))
            dq.append((w, s, 1))
        while dq:
            v, p, d = dq.popleft()
            # d is the current walk length; a return to s would be length d+1.
            # If d+1 >= g it cannot be a violation, so skip expansion.
            if d + 1 >= g:
                continue
            for w in adj[v]:
                if w == p:
                    continue
                if w == s:
                    return d + 1
                st = (w, v)
                if st not in seen:
                    seen.add(st)
                    dq.append((w, v, d + 1))
    return INF


def _girth_ok(adj, n, g):
    return _shortest_cycle(adj, n, g) >= g


# --------------------------------------------------------------------------- #
# LUW construction CD(d, q) -> connected component of the zero point          #
# --------------------------------------------------------------------------- #
def _luw_component(q, d):
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
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order = {start: 0}, [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


# --------------------------------------------------------------------------- #
# Cayley graph on Z_q x Z_2 with degree 5 (generators +/-a, +/-b, (0,1))      #
# --------------------------------------------------------------------------- #
def _cayley_zq_z2(q, a, b):
    n = 2 * q
    adj = [[] for _ in range(n)]
    edges = []
    for i in range(q):
        for t in (0, 1):
            u = 2 * i + t
            for (di, dt) in ((a, 0), (-a, 0), (b, 0), (-b, 0), (0, 1)):
                v = 2 * ((i + di) % q) + ((t + dt) & 1)
                if u < v:
                    adj[u].append(v)
                    adj[v].append(u)
                    edges.append((u, v))
    return n, adj, edges


def _is_prime(x):
    if x < 2:
        return False
    d = 2
    while d * d <= x:
        if x % d == 0:
            return False
        d += 1
    return True


def _next_prime(x):
    x = max(2, x)
    while not _is_prime(x):
        x += 1
    return x


# --------------------------------------------------------------------------- #
def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 120.0  # contract: 120 s per evaluation

    # -- guaranteed-valid baseline: LUW CD(d, q), d + 5 >= g ---------------- #
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if d + 5 < g:  # safety: nudge d up so girth bound covers g
        d += 1
    n_luw, edges_luw = _luw_component(q, d)
    best_n, best_edges = n_luw, edges_luw

    # -- attempt to beat the baseline with smaller verified-valid Cayley --- #
    # Moore bound (odd girth g = 2r+1, r=(g-1)//2): n >= 1 + d*(1+..+(d-1)^(r-1))
    r = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** e for e in range(r))
    lo_q = max(2, (moore + 1) // 2)
    # only candidates with n = 2q < best_n can improve
    hi_q = best_n // 2

    def _gen_pairs(q):
        # Generator pairs for Z_q x Z_2. Keep values small, distinct, nonzero,
        # and avoid a == b and a == -b (mod q) which create 2-cycles/relations.
        out = []
        seen = set()
        for a in range(1, min(12, q)):
            for b in range(1, min(12, q)):
                if b == a or b == (-a) % q or b == 0 or a == 0:
                    continue
                key = (a % q, b % q)
                if key in seen:
                    continue
                seen.add(key)
                out.append(key)
        return out

    q = lo_q
    while q <= hi_q and _time_left() > 0:
        q = _next_prime(q)
        for (a, b) in _gen_pairs(q):
            if q <= max(a, b):
                continue
            if _time_left() <= 0:
                break
            n, adj, edges = _cayley_zq_z2(q, a % q, b % q)
            if n < best_n and _girth_ok(adj, n, g):
                best_n, best_edges = n, edges
                # found a smaller valid one; continue scanning to shrink further
                # but we can stop once we hit the Moore bound (optimal in family)
                if 2 * q >= moore and n <= moore:
                    return n, edges
        q += 1

    return best_n, best_edges
