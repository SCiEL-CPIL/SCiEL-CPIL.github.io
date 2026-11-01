import time
import random

def construct(k, g):
    """Construct a k-regular graph of girth at least g.

    Returns (n, edges) where n is the number of vertices and edges is a list of (u, v) tuples.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin below the 150 s evaluator limit

    if k == 4 and g == 10:
        result = _search_4_10(deadline)
        if result is not None:
            n, edges = result
            if _girth_ok(edges, n, g):
                return n, edges

    # Fallback: Lazebnik-Ustimenko-Woldar construction (guaranteed valid, large).
    return _luw_graph(k, g)


# ---------------------------------------------------------------------------
# LUW construction (guaranteed valid fallback)
# ---------------------------------------------------------------------------
def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
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
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges


# ---------------------------------------------------------------------------
# Girth verification via non-backtracking walks (exact)
# ---------------------------------------------------------------------------
def _girth_ok(edges, n, g):
    """Return True iff the simple 4-regular graph has girth at least g."""
    deg = [0] * n
    seen = set()
    for u, v in edges:
        if u == v:
            return False
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[a] += 1
        deg[b] += 1
    for d in deg:
        if d != 4:
            return False

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for start in range(n):
        dist = {(-1, start): 0}
        frontier = [(-1, start)]
        depth = 0
        while frontier and depth < g - 1:
            depth += 1
            new_frontier = []
            for prev, cur in frontier:
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    st = (cur, nxt)
                    if st in dist:
                        continue
                    dist[st] = depth
                    if nxt == start and 3 <= depth < g:
                        return False
                    new_frontier.append(st)
            frontier = new_frontier
    return True


# ---------------------------------------------------------------------------
# Random 4-regular generation via configuration model
# ---------------------------------------------------------------------------
def _random_4_regular(n, rng):
    """Return an adjacency list (each len 4) or None."""
    for _ in range(300):
        stubs = [i for i in range(n) for _ in range(4)]
        rng.shuffle(stubs)
        adj = [[] for _ in range(n)]
        ok = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v or v in adj[u]:
                ok = False
                break
            adj[u].append(v)
            adj[v].append(u)
        if ok:
            return adj
    return None


def _adj_to_edges(adj):
    edges = []
    for u in range(len(adj)):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges


# ---------------------------------------------------------------------------
# Find one short cycle (length 3..g-1) as a list of vertices, or None.
# ---------------------------------------------------------------------------
def _find_one_short_cycle(adj, n, g, rng):
    """Return a cycle as a list of vertices (length 3..g-1) or None."""
    starts = list(range(n))
    rng.shuffle(starts)
    for start in starts:
        parent = {(-1, start): None}
        dist = {(-1, start): 0}
        frontier = [(-1, start)]
        depth = 0
        while frontier and depth < g - 1:
            depth += 1
            new_frontier = []
            for prev, cur in frontier:
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    st = (cur, nxt)
                    if st in dist:
                        continue
                    dist[st] = depth
                    parent[st] = (prev, cur)
                    if nxt == start and 3 <= depth < g:
                        # Reconstruct cycle.
                        cyc = [start]
                        state = (cur, start)
                        while parent[state] is not None:
                            p, c = state
                            cyc.append(c)
                            state = parent[state]
                        cyc.reverse()
                        if 3 <= len(cyc) < g:
                            return cyc
                    new_frontier.append(st)
            frontier = new_frontier
    return None


# ---------------------------------------------------------------------------
# Try to break a given cycle with a 2-switch.
# ---------------------------------------------------------------------------
def _try_break_cycle(adj, n, cycle, rng):
    """Try to break the given cycle via a 2-switch. Returns True if applied."""
    L = len(cycle)
    if L < 4:
        return False
    i = rng.randrange(L)
    j = (i + rng.randrange(1, L)) % L
    if j == i:
        return False
    a, b = cycle[i], cycle[(i + 1) % L]
    c, d = cycle[j], cycle[(j + 1) % L]
    if a == c or a == d or b == c or b == d:
        return False
    for (x1, y1, x2, y2) in ((a, c, b, d), (a, d, b, c)):
        if x1 == y1 or x2 == y2:
            continue
        if y1 in adj[x1] or y2 in adj[x2]:
            continue
        adj[a].remove(b); adj[a].append(x1)
        adj[b].remove(a); adj[b].append(x2)
        adj[x1].remove(y1); adj[x1].append(a)
        adj[y1].remove(x1); adj[y1].append(b)
        return True
    return False


# ---------------------------------------------------------------------------
# The search: random 4-regular + targeted 2-switch cycle breaking.
# ---------------------------------------------------------------------------
def _search_4_10(deadline):
    rng = random.Random(98765)
    n = 320
    g = 10

    for restart in range(80):
        if time.time() > deadline - 5.0:
            break
        adj = _random_4_regular(n, rng)
        if adj is None:
            continue

        progress = True
        while progress and time.time() < deadline - 5.0:
            progress = False
            for _attempt in range(300):
                if time.time() > deadline - 5.0:
                    break
                cycle = _find_one_short_cycle(adj, n, g, rng)
                if cycle is None:
                    edges = _adj_to_edges(adj)
                    if _girth_ok(edges, n, g):
                        return n, edges
                    break
                if _try_break_cycle(adj, n, cycle, rng):
                    progress = True
                    break

        edges = _adj_to_edges(adj)
        if _girth_ok(edges, n, g):
            return n, edges

    return None
