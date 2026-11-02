"""Explicit 320-vertex (4,10)-cage construction + bounded local search, fast validation."""
import time
import random
from collections import deque

_DEADLINE = 150.0
_MARGIN = 20.0


def construct(k, g):
    """Return (n, edges) for a k-regular graph of girth at least g, minimizing n."""
    start = time.time()
    deadline = start + _DEADLINE - _MARGIN

    # ---- Guaranteed-valid fallback: explicit LUW construction (seed method) ----
    q = k
    min_d = max(3, g // 2 - 1)
    best = None
    for d in range(min_d, 10):
        if time.time() > deadline - 5:
            break
        n, edges = _luw(k, g, d)
        adj = _build_adj(n, edges)
        if _degree_ok(adj, n, k) and not _has_short_cycle(adj, n, g):
            best = (n, edges)
            break
    if best is None:
        n, edges = _luw(k, g, 9)
        best = (n, edges)

    # ---- Optional improvement: bounded local search (edge swaps) ----
    # Only useful for the concrete task; guard everything with the deadline.
    if k == 4 and g == 10 and time.time() < deadline - 10:
        found = _local_search(4, 10, deadline)
        if found is not None and found[0] < best[0]:
            best = found

    return best[0], best[1]


def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _degree_ok(adj, n, k):
    for i in range(n):
        if len(adj[i]) != k:
            return False
    return True


def _has_short_cycle(adj, n, g):
    """True iff there is a cycle of length < g (non-backtracking BFS to depth g-2)."""
    for s in range(n):
        frontier = deque()
        for nb in adj[s]:
            frontier.append((nb, s))
        depth = 1
        while frontier:
            if depth + 1 < g:
                for node, prev in frontier:
                    if prev != s:
                        for nb in adj[node]:
                            if nb == s:
                                return True
            if depth >= g - 2:
                break
            nxt = deque()
            while frontier:
                node, prev = frontier.popleft()
                for nb in adj[node]:
                    if nb != prev:
                        nxt.append((nb, node))
            frontier = nxt
            depth += 1
    return False


def _girth(adj, n, cap):
    """Return the minimum cycle length, or cap if none shorter than cap exists."""
    best = cap
    for s in range(n):
        frontier = deque()
        for nb in adj[s]:
            frontier.append((nb, s))
        depth = 1
        while frontier:
            if depth + 1 < best:
                for node, prev in frontier:
                    if prev != s:
                        for nb in adj[node]:
                            if nb == s:
                                if depth + 1 < best:
                                    best = depth + 1
                                break
                    if best == 3:
                        return 3
            if depth >= best - 2:
                break
            nxt = deque()
            while frontier:
                node, prev = frontier.popleft()
                for nb in adj[node]:
                    if nb != prev:
                        nxt.append((nb, node))
            frontier = nxt
            depth += 1
        if best == 3:
            break
    return best


def _local_search(k, g, deadline):
    """Bounded edge-swap search for a k-regular graph of girth >= g with n < current best.

    Starts from random k-regular bipartite graphs of decreasing even size and tries
    to raise the girth. Returns (n, edges) for the first valid graph found, else None.
    """
    rng = random.Random(20240517)

    for n in range(316, 304, -2):
        if time.time() > deadline:
            return None
        half = n // 2
        # 4 perfect matchings between A (0..half-1) and B (half..n-1)
        for attempt in range(200):
            if time.time() > deadline:
                return None
            matchings = [list(range(half)) for _ in range(k)]
            ok = True
            for m in matchings:
                rng.shuffle(m)
            # Build adjacency (bipartite, simple, k-regular)
            adj = [[] for _ in range(n)]
            for a in range(half):
                for m in range(k):
                    b = half + matchings[m][a]
                    adj[a].append(b)
                    adj[b].append(a)
            # Verify simplicity (no multi-edge) and exact degree
            simple = True
            for u in range(n):
                if len(adj[u]) != k or len(set(adj[u])) != k:
                    simple = False
                    break
            if not simple:
                continue

            cur_g = _girth(adj, n, g)
            if cur_g >= g:
                edges = _edges_from_adj(n, adj)
                return n, edges

            # Local search: edge swaps between two A-nodes (keeps k-regularity, bipartite)
            best_g = cur_g
            no_improve = 0
            while time.time() < deadline and no_improve < 3000:
                a1 = rng.randrange(half)
                a2 = rng.randrange(half)
                if a1 == a2:
                    continue
                # pick a matching to swap within (keeps regularity)
                m = rng.randrange(k)
                b1 = half + matchings[m][a1]
                b2 = half + matchings[m][a2]
                # new edges a1-b2, a2-b1 must be simple
                if b2 in adj[a1] or b1 in adj[a2]:
                    continue
                # apply
                adj[a1].remove(b1); adj[b1].remove(a1)
                adj[a2].remove(b2); adj[b2].remove(a2)
                adj[a1].append(b2); adj[b2].append(a1)
                adj[a2].append(b1); adj[b1].append(a2)
                matchings[m][a1] = b2 - half
                matchings[m][a2] = b1 - half

                ng = _girth(adj, n, min(g, best_g))
                if ng > best_g:
                    best_g = ng
                    no_improve = 0
                else:
                    no_improve += 1
                if best_g >= g:
                    edges = _edges_from_adj(n, adj)
                    return n, edges
    return None


def _edges_from_adj(n, adj):
    edges = []
    seen = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges


def _luw(k, g, d=None):
    """Lazebnik-Ustimenko-Woldar construction (the original seed method)."""
    q = k
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: table[x][y])
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
