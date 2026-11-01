"""Cage construction: 4-regular (general k) graphs of given girth, minimized vertices."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

_START = time.time()
_DEADLINE = 130.0


def _elapsed():
    return time.time() - _START


def _time_ok():
    return _elapsed() < _DEADLINE


def _verify_simple_regular(edges, n, k):
    seen = set()
    deg = [0] * n
    for i, j in edges:
        if i < 0 or i >= n or j < 0 or j >= n or i == j:
            return False
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[a] += 1
        deg[b] += 1
    for d in deg:
        if d != k:
            return False
    return True


def _min_girth_bfs(n, edges, k):
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    best = 10**9
    for s in range(n):
        dist = {}
        dq = deque()
        for nb in adj[s]:
            dist[(nb, s)] = 1
            dq.append((nb, s))
        while dq:
            cur, prev = dq.popleft()
            d = dist[(cur, prev)]
            for nb in adj[cur]:
                if nb == prev:
                    continue
                key = (nb, cur)
                if key in dist:
                    L = d + 1
                    if L < best:
                        best = L
                else:
                    dist[key] = d + 1
                    dq.append((nb, cur))
        if best <= k:
            break
    return best if best < 10**9 else 10**9


def _luzebnik_ustimenko_woldar(k, g):
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
        if not _time_ok():
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _voltage_graph_4_10():
    """Voltage graph lift over Z_80 with 4-vertex base (C_4 with doubled edges)."""
    n_base = 4
    base_edges = [
        (0, 1, 0), (0, 1, 1),
        (1, 2, 0), (1, 2, 1),
        (2, 3, 0), (2, 3, 1),
        (3, 0, 0), (3, 0, 1),
    ]
    m = 80
    n = n_base * m

    # Try several voltage assignments
    assignments = [
        {0: 1, 1: 3, 2: 5, 3: 7, 4: 9, 5: 11, 6: 13, 7: 15},
        {0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 32, 6: 5, 7: 10},
        {0: 7, 1: 14, 2: 21, 3: 28, 4: 35, 5: 42, 6: 49, 7: 56},
    ]

    for volt in assignments:
        edges = []
        for v in range(n_base):
            for g in range(m):
                for idx, (u, w, ei) in enumerate(base_edges):
                    if u == v:
                        g2 = (g + volt[ei]) % m
                        edges.append((v * m + g, w * m + g2))
                for idx, (u, w, ei) in enumerate(base_edges):
                    if w == v:
                        g2 = (g - volt[ei]) % m
                        edges.append((v * m + g, u * m + g2))

        seen = set()
        unique_edges = []
        for i, j in edges:
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in seen:
                seen.add((a, b))
                unique_edges.append((a, b))

        deg = [0] * n
        for i, j in unique_edges:
            deg[i] += 1
            deg[j] += 1

        if len(unique_edges) != n * 2:
            continue
        if min(deg) != 4 or max(deg) != 4:
            continue

        if _min_girth_bfs(n, unique_edges, 4) >= 10:
            return n, unique_edges

    return None, None


def _edge_switch_search_4_10():
    """
    Local edge-switching (2-switch) search to improve girth.
    Start from L-U-W graph, try 2-switches that increase girth.
    """
    n, edges = _luzebnik_ustimenko_woldar(4, 10)
    if not _verify_simple_regular(edges, n, 4):
        return None, None

    # Build adjacency
    adj = [[] for _ in range(n)]
    edge_set = set()
    for i, j in edges:
        a, b = (i, j) if i < j else (j, i)
        adj[i].append(j)
        adj[j].append(i)
        edge_set.add((a, b))

    current_girth = _min_girth_bfs(n, edges, 4)
    if current_girth >= 10:
        return n, edges

    best_girth = current_girth
    best_edges = list(edges)
    best_adj = [list(a) for a in adj]
    best_edge_set = set(edge_set)

    # Collect all edges as list for random sampling
    edge_list = list(edge_set)

    # Try random 2-switches
    max_iters = 50000
    rng = random.Random(42)
    for it in range(max_iters):
        if not _time_ok():
            break

        # Pick two random edges
        e1 = rng.choice(edge_list)
        e2 = rng.choice(edge_list)
        if e1 == e2:
            continue

        a1, b1 = e1
        a2, b2 = e2

        # Check if switching would create a loop or duplicate edge
        # Switch: remove (a1,b1) and (a2,b2), add (a1,a2) and (b1,b2)
        # or (a1,b2) and (b1,a2)
        for new_e1, new_e2 in [
            (a1, a2), (b1, b2)
        ], [
            (a1, b2), (b1, a2)
        ]:
            if new_e1 == new_e2:
                continue
            na1, nb1 = (new_e1, ) if new_e1 < new_e2 else (new_e2,)
            na2, nb2 = (new_e2, ) if new_e1 < new_e2 else (new_e1,)

            # Normalize
            ne1 = (min(new_e1, new_e2), max(new_e1, new_e2))
            ne2 = (min(new_e2, new_e1), max(new_e2, new_e1))

            if ne1 in edge_set or ne2 in edge_set:
                continue
            if ne1 == ne2:
                continue

            # Apply switch
            edge_set.discard(e1)
            edge_set.discard(e2)
            edge_set.add(ne1)
            edge_set.add(ne2)

            # Rebuild adjacency incrementally
            adj[a1].remove(b1)
            adj[b1].remove(a1)
            adj[a2].remove(b2)
            adj[b2].remove(a2)

            adj[a1].append(a2 if ne1[0] == a1 else b1)
            # Actually this is getting complex. Let me rebuild adj from edge_set.
            # For efficiency, let me just check girth on the new edge set.

            # Rebuild adj
            new_adj = [[] for _ in range(n)]
            for i, j in edge_set:
                new_adj[i].append(j)
                new_adj[j].append(i)

            new_edges = list(edge_set)
            g = _min_girth_bfs(n, new_edges, 4)

            if g > best_girth:
                best_girth = g
                best_edges = new_edges
                best_adj = new_adj
                best_edge_set = set(edge_set)
                edge_list = list(edge_set)
                if best_girth >= 10:
                    return n, best_edges

            # Undo
            edge_set.discard(ne1)
            edge_set.discard(ne2)
            edge_set.add(e1)
            edge_set.add(e2)

    return n, best_edges if best_girth >= 10 else (None, None)


def _petersen():
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (1, 3), (1, 4),
        (5, 6), (5, 7), (5, 8), (5, 9),
        (6, 7), (6, 8), (6, 9),
        (7, 8), (7, 9), (8, 9),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    return 10, edges


def _hoffman_singleton():
    n = 50
    edges = []
    def idx(a, b):
        return a * 10 + b
    for a in range(5):
        for b in range(10):
            v = idx(a, b)
            nbs = [
                idx(a, (b + 1) % 10), idx(a, (b - 1) % 10),
                idx((a + 1) % 5, (2 * b) % 10), idx((a + 1) % 5, (2 * b + 1) % 10),
                idx((a + 2) % 5, (4 * b) % 10), idx((a + 2) % 5, (4 * b + 1) % 10),
            ]
            for w in nbs:
                if v < w:
                    edges.append((v, w))
    return n, edges


def _moore_girth5(k):
    known = {3: 10, 4: 14, 5: 30, 6: 42, 7: 50}
    if k in known:
        if k == 7:
            return _hoffman_singleton()
        if k == 3:
            return _petersen()
    return _luzebnik_ustimenko_woldar(k, 5)


def construct(k, g):
    """Return (n, edges) for a k-regular graph of girth >= g, minimizing n."""
    if k == 3 and g == 5:
        return _petersen()
    if k == 7 and g == 5:
        return _hoffman_singleton()
    if g == 5:
        return _moore_girth5(k)
    if k == 4 and g == 10:
        # Try voltage graph first
        n, edges = _voltage_graph_4_10()
        if n is not None:
            return n, edges
        # Fall back to L-U-W
        n, edges = _luzebnik_ustimenko_woldar(k, g)
        return n, edges

    # General: L-U-W construction
    n, edges = _luzebnik_ustimenko_woldar(k, g)
    return n, edges
# EVOLVE-BLOCK-END
