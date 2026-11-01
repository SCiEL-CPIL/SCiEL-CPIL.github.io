"""Construct a 4-regular graph of girth 10 via Lazebnik-Ustimenko-Woldar CD(d, q)."""
import time
from collections import deque

# EVOLVE-BLOCK-START
def _build_luw(q, d, deadline, max_vertices=None):
    """Build the connected component of the zero point in CD(d, q).
    Returns (n, edges) or None if the deadline is exceeded / size cap hit."""
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
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        if time.time() > deadline:
            return None
        if max_vertices is not None and len(order) > max_vertices:
            return None
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                if max_vertices is not None and len(order) > max_vertices:
                    return None
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _verify(n, edges, k, g, deadline):
    """Return True iff the graph is simple, k-regular, and has girth >= g.
    Girth is certified exactly with non-backtracking walks from every vertex."""
    if time.time() > deadline:
        return False
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        if v in adj[u]:  # repeated / self edge
            return False
        adj[u].add(v)
        adj[v].add(u)
    for u in range(n):
        if len(adj[u]) != k:
            return False

    # Non-backtracking BFS from every vertex; a cycle of length L < g is found
    # when we reach the start within < g steps along a non-backtracking walk.
    # Optimization: use a list as queue for speed, and early-exit check.
    for s in range(n):
        dist = {}
        queue = []
        for nb in adj[s]:
            queue.append((nb, s, 1))
            dist[(nb, s)] = 1
        qi = 0
        while qi < len(queue):
            cur, prev, d = queue[qi]
            qi += 1
            if cur == s and d < g:
                return False
            if d >= g - 1:
                continue
            for nb in adj[cur]:
                if nb == prev:
                    continue
                st = (nb, cur)
                if st not in dist:
                    dist[st] = d + 1
                    queue.append((nb, cur, d + 1))
        if time.time() > deadline:
            return False
    return True


def construct(k, g):
    start_time = time.time()
    hard_deadline = start_time + 118.0

    q = k
    # Candidate d values: the L-U-W construction guarantees girth >= d + 5 (odd d).
    # A larger d can still be valid (girth may exceed d+5) while yielding a smaller
    # connected component for small q. Try candidates ordered by estimated size.
    base = g - 5
    if base % 2 == 0:
        base += 1
    base = max(3, base)

    # Estimate connected-component size ~ 2 * q * q^{d-1} (each side has q^d points).
    def est(d):
        return 2 * q * (q ** (d - 1))

    candidates = []
    for d in range(base, base + 8):
        if d % 2 == 1:
            candidates.append(d)
    candidates.sort(key=lambda d: est(d))

    best = None
    for d in candidates:
        if time.time() > hard_deadline - 4.0:
            break
        # Time budget for building + verifying this candidate.
        build_deadline = min(hard_deadline - 6.0, time.time() + 30.0)
        res = _build_luw(q, d, build_deadline)
        if res is None:
            continue
        n, edges = res
        if n % 2 != 0:
            # not a valid bipartite regular component; skip
            continue
        # Verify girth (and regularity/simplicity).
        ok = _verify(n, edges, k, g, hard_deadline - 2.0)
        if ok:
            if best is None or n < best[0]:
                best = (n, edges)
            # found a valid graph; stop searching larger d (they are bigger)
            break

    if best is not None:
        return best

    # Fallback: the seed's fixed construction (guaranteed valid by theory).
    d = base
    res = _build_luw(q, d, hard_deadline - 1.0)
    if res is not None:
        return res

    # Last resort: build with a very large d to guarantee girth, if any time remains.
    for d in range(base, base + 4):
        if d % 2 == 1:
            res = _build_luw(q, d, hard_deadline - 0.5)
            if res is not None:
                return res
    raise ValueError("Failed to construct a valid graph within the time budget")
# EVOLVE-BLOCK-END
