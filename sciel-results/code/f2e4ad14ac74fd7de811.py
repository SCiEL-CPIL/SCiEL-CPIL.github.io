"""Structural: algebraic/circulant construction for small girth-10 4-regular
graphs, with the Lazebnik-Ustimenko-Woldar graph as a guaranteed-valid fallback."""
import time
import random
import math

_DEADLINE = 150.0          # evaluator wall-clock limit
_SAFETY = 10.0             # safety margin before the limit
_START = None

def _now():
    return time.time()

def _time_left():
    if _START is None:
        return _DEADLINE
    return _DEADLINE - _SAFETY - (_now() - _START)


def _girth_fast(adj, n, limit):
    """Return min cycle length via non-backtracking BFS from each vertex,
    capped at 'limit' (early exit once we know girth >= limit)."""
    from collections import deque
    best = limit + 1
    for s in range(n):
        seen = {}
        q = deque()
        for nb in adj[s]:
            seen[(s, nb)] = 1
            q.append((s, nb, 1))
        while q:
            prev, v, dist = q.popleft()
            if dist + 1 >= best:
                continue
            for nb in adj[v]:
                if nb == prev:
                    continue
                if nb == s and dist + 1 >= 3:
                    cyc = dist + 1
                    if cyc < best:
                        best = cyc
                    if best <= limit:
                        return best
                    continue
                key = (v, nb)
                if key in seen:
                    cyc = seen[key] + dist + 1
                    if cyc < best:
                        best = cyc
                    if best <= limit:
                        return best
                    continue
                seen[key] = dist + 1
                q.append((v, nb, dist + 1))
    return best


# ---------------------------------------------------------------------------
# Girth certification via non-backtracking walks from every vertex.
# Returns the girth (minimum cycle length). A graph with no cycle < g has
# girth >= g (or is a tree, but a 4-regular graph always has cycles).
# ---------------------------------------------------------------------------
def _girth(adj, n):
    """adj: list of sorted neighbor lists. Returns girth (int)."""
    best = 10 ** 9
    for s in range(n):
        # non-backtracking BFS: state (v, prev)
        from collections import deque
        seen = {}
        q = deque()
        for nb in adj[s]:
            seen[(s, nb)] = 1
            q.append((s, nb, 1))
        while q:
            prev, v, dist = q.popleft()
            for nb in adj[v]:
                if nb == prev:
                    continue
                if nb == s and dist + 1 >= 3:
                    cyc = dist + 1
                    if cyc < best:
                        best = cyc
                    # do not expand further from a return to s
                    continue
                key = (v, nb)
                if key in seen:
                    cyc = seen[key] + dist + 1
                    if cyc < best:
                        best = cyc
                    continue
                seen[key] = dist + 1
                q.append((v, nb, dist + 1))
    return best


def _edges_from_adj(adj):
    edges = []
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges


# ---------------------------------------------------------------------------
# Algebraic constructions: circulant / generalized Petersen style.
# ---------------------------------------------------------------------------
def _circulant_adj(n, steps):
    """4-regular circulant: each vertex i connected to i±a, i±b (mod n)."""
    a, b = steps
    adj = [[] for _ in range(n)]
    for i in range(n):
        for s in (a, -a, b, -b):
            j = (i + s) % n
            if j != i and j not in adj[i]:
                adj[i].append(j)
    for i in range(n):
        if len(adj[i]) != 4:
            return None
        adj[i].sort()
    return adj


def _gen_petersen_adj(n, step):
    """Generalized Petersen-like: outer cycle i~i±1, plus chords i~(i+step)%n.
    For n even this gives degree 4: neighbors i-1, i+1, i+step, i-step."""
    if n < 5:
        return None
    adj = [[] for _ in range(n)]
    for i in range(n):
        for s in (1, -1, step, -step):
            j = (i + s) % n
            if j != i and j not in adj[i]:
                adj[i].append(j)
    for i in range(n):
        if len(adj[i]) != 4:
            return None
        adj[i].sort()
    return adj


def _algebraic_search(k, g, time_budget):
    """Try a family of algebraic 4-regular graphs, return best valid (n, edges)."""
    t0 = _now()
    best = None
    # Candidate n values between Moore bound and a cap
    moore = _moore_bound(k, g)
    max_n = 400
    for n in range(max(moore, 20), max_n + 1):
        if _now() - t0 > time_budget or _time_left() < 15:
            break
        # Try circulant with various step pairs
        found = False
        for a in range(1, n // 2 + 1):
            if _now() - t0 > time_budget or _time_left() < 15:
                break
            for b in range(a + 1, n // 2 + 1):
                if 2 * a == n or 2 * b == n:
                    continue  # would create duplicate edges
                adj = _circulant_adj(n, (a, b))
                if adj is None:
                    continue
                gf = _girth_fast(adj, n, g)
                if gf >= g:
                    edges = _edges_from_adj(adj)
                    if best is None or n < best[0]:
                        best = (n, edges)
                    found = True
                    break
            if found:
                break
        # Also try generalized Petersen
        if not found:
            for step in range(2, n // 2):
                if _now() - t0 > time_budget or _time_left() < 15:
                    break
                if 2 * step == n:
                    continue
                adj = _gen_petersen_adj(n, step)
                if adj is None:
                    continue
                gf = _girth_fast(adj, n, g)
                if gf >= g:
                    edges = _edges_from_adj(adj)
                    if best is None or n < best[0]:
                        best = (n, edges)
                    break
    return best


# ---------------------------------------------------------------------------
# Lazebnik-Ustimenko-Woldar fallback (valid, large).
# ---------------------------------------------------------------------------
def _luw(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
# Moore bound for (k, g) with g even: 1 + k * sum_{i=0}^{g/2 - 2} (k-1)^i
# ---------------------------------------------------------------------------
def _moore_bound(k, g):
    if g % 2 == 0:
        m = g // 2
        return 1 + k * sum((k - 1) ** i for i in range(m - 1))
    else:
        m = (g - 1) // 2
        return k * sum((k - 1) ** i for i in range(m))


def construct(k, g):
    global _START
    _START = _now()

    # Structural search: algebraic/circulant constructions
    budget = min(100.0, _time_left())
    best_result = _algebraic_search(k, g, budget)

    if best_result is not None:
        return best_result

    # Fallback: LUW graph (valid, large)
    return _luw(k, g)
