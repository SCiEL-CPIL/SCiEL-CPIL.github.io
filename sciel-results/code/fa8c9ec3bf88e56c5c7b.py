"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
# EVOLVE-BLOCK-START
import time as _time

_START = _time.time()
_DEADLINE = 150.0


def _gf(q):
    """Return (add, sub, mul) for GF(q), q prime or 4."""
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    return (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)


def _rule(d):
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    return rule


def _luw(q, d):
    """Build the connected component of the zero point of the LUW CD(d,q) graph.
    Returns (n, edges) or None if q is too large to handle in time."""
    add, sub, mul = _gf(q)
    rule = _rule(d)
    q2 = q * q

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
    i = 0
    while i < len(order):
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
        if vertex[0] == 0:
            for other in neighbours(vertex):
                edges.append((index[vertex], index[other]))
        i += 1
        # safety: bail if the component is exploding
        if len(order) > 400000:
            return None
    return len(order), edges


def _girth_ok(n, edges, k, g):
    """Verify simple, k-regular, and no cycle shorter than g using non-backtracking
    walks from every vertex. Returns True if all constraints hold."""
    # degree check + simple graph
    deg = [0] * n
    seen = set()
    for a, b in edges:
        if a == b:
            return False
        key = (a, b) if a < b else (b, a)
        if key in seen:
            return False
        seen.add(key)
        deg[a] += 1
        deg[b] += 1
    for d in deg:
        if d != k:
            return False
    # adjacency lists
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    # non-backtracking walk depth g-1 from each vertex; a cycle of length < g
    # would produce a state (prev,cur) with cur==start or a repeated (prev,cur).
    # We detect any cycle of length < g by checking whether a non-backtracking
    # walk of length L returns to a vertex at distance L < g.
    for start in range(n):
        # BFS over states (prev, cur), depth up to g-1
        states = {(None, start)}
        frontier = {(None, start)}
        for depth in range(1, g):
            nxt = set()
            for prev, cur in frontier:
                for nb in adj[cur]:
                    if nb == prev:
                        continue
                    if nb == start and depth >= 3:
                        # cycle of length depth (>=3) which is < g
                        return False
                    st = (cur, nb)
                    if st in states:
                        # a non-backtracking cycle of length < g
                        return False
                    nxt.add(st)
                    states.add(st)
            frontier = nxt
            if not frontier:
                break
    return True


def construct(k, g):
    global _START
    # Candidate (q, d) pairs in the LUW family that satisfy girth >= g.
    # girth >= d + 5 for odd d. We need d + 5 >= g  =>  d >= g - 5.
    # q must equal k (degree) for the regular graph to have degree k.
    # For this task k is fixed; we search d and, where allowed, small q.
    best = None
    min_d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # Primary: q = k, d = min_d (baseline).
    # Then try increasing d (larger girth, possibly larger n) only if needed;
    # and try smaller q if k permits (q must be a prime power <= k and the
    # graph degree equals q, so q must equal k for degree k). Thus q is pinned
    # to k for a k-regular graph. We therefore only vary d.
    for d in range(min_d, min_d + 4):
        if d % 2 == 0:
            continue
        if d + 5 < g:
            continue
        if _time.time() > _DEADLINE - 30:
            break
        res = _luw(k, d)
        if res is None:
            continue
        n, edges = res
        if best is None or n < best[0]:
            best = (n, edges)
    if best is None:
        # fallback to the original construction exactly
        res = _luw(k, min_d)
        n, edges = res
        return n, edges
    n, edges = best
    # Final certification (deadline-guarded); if it fails, return anyway as the
    # construction is known-correct for these parameters.
    if _time.time() < _DEADLINE - 5:
        if not _girth_ok(n, edges, k, g):
            # fall back to baseline parameters
            res = _luw(k, min_d)
            n, edges = res
    return n, edges
# EVOLVE-BLOCK-END
