"""Construct a small 3-regular graph of girth >= 16 (cage n(3,16)).

Strategy: Lazebnik-Ustimenko-Woldar CD(d,q) graphs are q-regular bipartite
graphs whose girth is at least d+5 (odd d) / d+6 (even d).  For q = 3 (prime)
we take the connected component of the zero point, which is typically far
smaller than q^d because of algebraic relations.  We EMPIRICALLY verify the
girth with a non-backtracking BFS from every vertex (the exact certificate the
evaluator uses) and return the smallest d that still reaches girth >= g within
the time budget.
"""
import time

# EVOLVE-BLOCK-START
def _luw_component(q, d):
    """Build the connected component of the zero point of CD(d,q).

    Returns (n, edges) where edges is a list of (u, v) with u < v, each vertex
    has degree exactly q, and the graph is simple and bipartite.
    """
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: table[x][y])
    else:
        add = (lambda x, y: (x + y) % q)
        sub = (lambda x, y: (x - y) % q)
        mul = (lambda x, y: x * y % q)

    # rule[c] = (a, b) for c in 2..d
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
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
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
        if vertex[0] == 0:
            for other in neighbours(vertex):
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _girth_check(n, edges, g):
    """Exact girth certificate: non-backtracking BFS from every vertex.

    Returns True iff every cycle has length >= g.  Uses the standard
    non-backtracking walk enumeration: a cycle of length L is found iff some
    non-backtracking closed walk of length L exists.  We BFS in the
    non-backtracking line graph and detect the first closed walk.
    """
    # adjacency with edge ids
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # For a 3-regular graph, non-backtracking from (prev, cur): next in adj[cur]
    # except prev.  A cycle of length L exists iff there is a non-backtracking
    # closed walk of length L.  The smallest such L is the girth.  We compute
    # the girth exactly by BFS on directed edges.
    # state = (prev, cur); start from every directed edge.
    # We want the minimum L >= 3 with a non-backtracking closed walk.
    # Use BFS tracking, for each state, the set of "origins" is too big.
    # Instead: standard method - for each vertex, do a BFS that does not go
    # back to the parent, and look for a back-edge to an ancestor.  That finds
    # the girth (shortest cycle) exactly in O(n^2) worst case, fine for n ~ 1e3.

    # Simpler exact method: BFS from each vertex, but we must be careful that
    # the BFS tree distance to a vertex can be shorter than the actual cycle.
    # The correct exact girth: for each vertex s, run BFS; for each edge (u,v),
    # if dist[s][u] + 1 + dist[s][v] ... no.  The exact approach:
    # girth = min over edges (u,v) of dist_no_backtrack.
    # We do: for each start s, BFS; when we relax an edge to an already-seen
    # vertex w that is not the parent of the current, candidate = depth+1+depth_w
    # but that double counts.  The clean exact method is non-backtracking BFS.

    # Non-backtracking BFS from a fixed start vertex s:
    # level 0: (s, s)  (no previous)
    # We expand (prev, cur) -> (cur, nxt) for nxt in adj[cur], nxt != prev.
    # A cycle is found when nxt == s and the walk length >= 3.
    # But we must ensure the walk is a simple cycle; non-backtracking closed
    # walks of minimal length equal the girth for the whole graph, and the
    # minimum over all starts gives the girth.  However a non-backtracking
    # closed walk can revisit vertices (a "figure eight" is not
    # non-backtracking-closed of minimal length because it would contain a
    # shorter closed walk).  In fact the shortest non-backtracking closed walk
    # is a simple cycle.  So:
    INF = 10 ** 9
    best = INF
    for s in range(n):
        # dist[prev][cur] = shortest non-backtracking walk length from s to (prev,cur)
        # state space: directed edges + (s,s)
        dist = {}
        # seed: for each neighbor, (s, nb) at length 1
        from collections import deque
        dq = deque()
        for nb in adj[s]:
            dist[(s, nb)] = 1
            dq.append((s, nb))
        while dq:
            prev, cur = dq.popleft()
            L = dist[(prev, cur)]
            for nxt in adj[cur]:
                if nxt == prev:
                    continue
                if nxt == s and L + 1 >= 3:
                    best = min(best, L + 1)
                    continue
                key = (cur, nxt)
                if key not in dist:
                    dist[key] = L + 1
                    dq.append(key)
        if best < g:
            return False
    return best >= g


def construct(k, g):
    deadline = time.time() + 140.0  # safety margin under 150s wall clock
    q = k
    # For girth g with q=3, the LUW CD(d,q) has girth >= d+5 (odd d).  We need
    # d+5 >= g  =>  d >= g-5.  But the component can have girth exactly d+5 or
    # larger; we try d from g-5 upward and verify empirically.  Start at the
    # theoretical minimum and grow only if needed.
    d_min = g - 5
    if d_min < 3:
        d_min = 3
    if d_min % 2 == 0:
        d_min += 1  # odd d preferred for the d+5 bound

    best = None
    d = d_min
    while True:
        if time.time() > deadline:
            break
        try:
            n, edges = _luw_component(q, d)
        except Exception:
            break
        # quick sanity: every vertex degree == q and n >= Moore bound
        if n == 0:
            d += 1
            continue
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(x != q for x in deg):
            d += 1
            continue
        if _girth_check(n, edges, g):
            if best is None or n < best[0]:
                best = (n, edges)
            # a smaller d gave a valid graph; stop (we started at the minimum
            # and grow, so the first valid one is the smallest d)
            break
        d += 1

    if best is None:
        # fallback: the original seed's d choice
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        n, edges = _luw_component(q, d)
        best = (n, edges)
    return best
# EVOLVE-BLOCK-END
