"""Rewritten program: algebraic (LUW) constructions with optimized girth check."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
import math as _math
from collections import deque

def _verify_girth_fast(adj, n, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex."""
    for start in range(n):
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _luw_construct(q, g):
    """LUW construction for (q+1)-regular graphs. Returns (n, edges)."""
    if q == 3:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
    elif q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
    else:
        p = q
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        add = lambda x, y: (x + y) % p
        sub = lambda x, y: (x - y) % p
        mul = lambda x, y: x * y % p
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(p):
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

def _moore_bound(k, g):
    """Compute the Moore bound for a k-regular graph of girth g."""
    if g % 2 == 0:
        # Even girth: n >= 2 * (k-1) * sum_{i=0}^{g/2-1} (k-1)^i
        half = g // 2
        s = 0
        term = 1
        for i in range(half):
            s += term
            term *= (k - 1)
        return 2 * s
    else:
        # Odd girth: n >= (k-1) * sum_{i=0}^{(g-1)/2} (k-1)^i + sum_{i=0}^{(g-3)/2} (k-1)^i
        h1 = (g - 1) // 2
        h2 = (g - 3) // 2
        s1 = 0
        term = 1
        for i in range(h1 + 1):
            s1 += term
            term *= (k - 1)
        s2 = 0
        term = 1
        for i in range(h2 + 1):
            s2 += term
            term *= (k - 1)
        return (k - 1) * s1 + s2

def _build_voltage_lift(base_edges, m, voltages):
    """Build a voltage graph lift. Returns (n, adj_list, edges) or None."""
    base_n = len(set([u for u, v in base_edges] + [v for u, v in base_edges]))
    n = base_n * m
    adj = [set() for _ in range(n)]
    edges = []
    for u, v in base_edges:
        vol = voltages.get((u, v), 0)
        for i in range(m):
            a = u * m + i
            b = v * m + (i + vol) % m
            if b not in adj[a]:
                adj[a].add(b)
                adj[b].add(a)
                edges.append((a, b))
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    adj_list = [list(s) for s in adj]
    return n, adj_list, edges

def _girth_check_adj(adj_list, n, g):
    """Non-backtracking BFS girth check using adjacency lists."""
    for start in range(n):
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj_list[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 140

    if k != 3:
        n, edges = _luw_construct(k, g)
        return n, edges

    # LUW construction is the primary and most reliable algebraic method
    n_luw, edges_luw = _luw_construct(3, g)
    best_n = n_luw
    best_edges = edges_luw

    # Try to improve with voltage lifts of small base graphs
    # K3,3 base (6 vertices, 9 edges, 3-regular)
    k33_edges = [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]
    # Petersen base (10 vertices, 15 edges, 3-regular)
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]

    # Try deterministic voltage assignments for small m
    # For K3,3: assign voltages to 9 edges
    # For Petersen: assign voltages to 15 edges
    # Use small m values and try a few deterministic patterns

    if _time.time() < deadline - 30:
        # Try K3,3 lifts with m from small values
        for m in range(2, min(30, best_n // 6 + 1)):
            if _time.time() > deadline - 15:
                break
            n_lift = 6 * m
            if n_lift >= best_n:
                continue

            # Try a few deterministic voltage patterns
            patterns = [
                [0, 1, 2, 3, 4, 5, 6, 7, 8],  # sequential
                [0, 1, 0, 1, 0, 1, 0, 1, 0],  # alternating
                [i % m for i in range(9)],     # modular
                [1] * 9,  # all ones
                [m // 2] * 9,  # half
            ]
            for pat in patterns:
                if _time.time() > deadline - 15:
                    break
                voltages = {}
                for i, (u, v) in enumerate(k33_edges):
                    voltages[(u, v)] = pat[i] % m
                    voltages[(v, u)] = (-pat[i]) % m
                result = _build_voltage_lift(k33_edges, m, voltages)
                if result is not None:
                    n_r, adj_r, edges_r = result
                    if _girth_check_adj(adj_r, n_r, g):
                        if n_r < best_n:
                            best_n = n_r
                            best_edges = edges_r
                        break

    if _time.time() < deadline - 30:
        # Try Petersen lifts
        for m in range(2, min(25, best_n // 10 + 1)):
            if _time.time() > deadline - 15:
                break
            n_lift = 10 * m
            if n_lift >= best_n:
                continue

            patterns = [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                [i % m for i in range(15)],
                [1] * 15,
            ]
            for pat in patterns:
                if _time.time() > deadline - 15:
                    break
                voltages = {}
                for i, (u, v) in enumerate(petersen_edges):
                    voltages[(u, v)] = pat[i] % m
                    voltages[(v, u)] = (-pat[i]) % m
                result = _build_voltage_lift(petersen_edges, m, voltages)
                if result is not None:
                    n_r, adj_r, edges_r = result
                    if _girth_check_adj(adj_r, n_r, g):
                        if n_r < best_n:
                            best_n = n_r
                            best_edges = edges_r
                        break

    return best_n, best_edges
# EVOLVE-BLOCK-END
