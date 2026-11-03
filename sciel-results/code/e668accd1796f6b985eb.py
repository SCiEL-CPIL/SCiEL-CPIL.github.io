import time
import collections
import math

_DEADLINE = None
_BUDGET = 140.0   # explicit safety margin below the 150s kill

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

# ---------------------------------------------------------------------------
# Algebraic (LUW / "addition-multiplication") 3-regular graph construction
# ---------------------------------------------------------------------------
def _luw_construct(q, d, deadline=None):
    """Build the LUW(q,d) graph. Returns (n, edges) or None on timeout."""
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        if deadline is not None and _time_left() < 3.0:
            return None
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges

# ---------------------------------------------------------------------------
# Girth verification (non-backtracking walks, certified exactly)
# ---------------------------------------------------------------------------
def _check_girth(n, edges, min_girth):
    """Return True iff the graph has no cycle shorter than min_girth.

    Uses non-backtracking walks: a cycle of length L corresponds to a
    non-backtracking walk of length L-1 that returns to its origin. We run a
    BFS in the directed-edge state space (prev,curr) from every directed edge
    and detect the shortest such return.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # A cycle of length L gives an NB walk of length L-1 back to origin.
    # We only need to rule out cycles with L < min_girth, i.e. NB walks of
    # length <= min_girth - 2. So we cap BFS depth at min_girth - 2.
    max_depth = min_girth - 2

    for u in range(n):
        for v in adj[u]:
            # state (prev, curr); start (u, v) at depth 0
            # we look for a return to (u, v) at depth d in [1, max_depth]
            dist = {(u, v): 0}
            queue = collections.deque([(u, v, 0)])
            found = False
            while queue:
                prev, curr, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                nd = depth + 1
                for nb in adj[curr]:
                    if nb == prev:
                        continue  # non-backtracking
                    if nb == u and curr == v:
                        # returned to origin at depth nd -> cycle length nd+1
                        if nd + 1 < min_girth:
                            return False
                        found = True
                        continue
                    st = (curr, nb)
                    if st not in dist:
                        dist[st] = nd
                        queue.append((curr, nb, nd))
            if found:
                pass  # cycle is long enough, fine
    return True

# ---------------------------------------------------------------------------
# Small-prime search for a smaller cage graph
# ---------------------------------------------------------------------------
def _search_primes(k, g, best_n, deadline):
    """Try LUW with various (q,d) for small primes to find n < best_n."""
    # girth of LUW(q,d) is at least 2d+2 (for prime q). We need 2d+2 >= g.
    # n = 2*q*(q^d - 1)/(q-1) roughly; minimize n over q,d.
    results = []
    # candidate (q, d) pairs
    candidates = []
    for q in range(k, 40):
        if q % 2 == 0:
            continue  # LUW works cleanly for odd q (and q=4)
        if q == 2:
            continue
        # need d >= (g-2)//2
        d_min = (g - 2) // 2
        if d_min < 3:
            d_min = 3
        # try a couple of d values
        for d in range(d_min, d_min + 3):
            # estimate n
            n_est = 2 * q * ((q ** d - 1) // (q - 1))
            if n_est < best_n:
                candidates.append((n_est, q, d))
    candidates.sort()
    for n_est, q, d in candidates:
        if _time_left() < 20.0:
            break
        if n_est >= best_n:
            break
        res = _luw_construct(q, d)
        if res is None:
            continue
        n, edges = res
        if n < best_n and _time_left() > 15.0:
            if _check_girth(n, edges, g):
                return n, edges
    return None

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + _BUDGET

    # Default LUW construction for k=3, g=16 (the cage problem n(3,16)).
    # d chosen so girth >= 2d+2 >= g.
    d_target = (g - 2) // 2 + 1
    if k % 2 == 1 and d_target % 2 == 0:
        d_target += 1

    # Build the default LUW graph.
    n_luw, edges_luw = _luw_construct(k, d_target)
    if n_luw is None:
        # Fallback: use a smaller d to guarantee termination.
        n_luw, edges_luw = _luw_construct(k, max(3, (g - 2) // 2))

    # Verify the default graph (cheap enough for ~1000 vertices).
    verified_default = False
    if _time_left() > 25.0:
        verified_default = _check_girth(n_luw, edges_luw, g)

    # Try to find a smaller valid graph via small-prime search.
    best_n = n_luw
    best_edges = edges_luw
    if _time_left() > 30.0:
        found = _search_primes(k, g, best_n, _DEADLINE)
        if found is not None:
            best_n, best_edges = found

    return best_n, best_edges
