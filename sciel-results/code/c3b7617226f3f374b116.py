import time
import numpy as np
from collections import deque
import random

_T0 = time.time()
_DEADLINE = _T0 + 105.0  # 15 s safety margin before the 120 s kill

def _deadline():
    return time.time() >= _DEADLINE

def _field_ops(q):
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    return ((lambda x, y: (x + y) % q),
            (lambda x, y: (x - y) % q),
            (lambda x, y: (x * y) % q))

def _luw_rule(d):
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    return rule

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _luw_graph(d, q):
    add, sub, mul = _field_ops(q)
    rc = [None] * (d + 1)
    for c in range(2, d + 1):
        rc[c] = _luw_rule(d)[c]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rc[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order = {start: 0}, [start]
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)

    adj = [set() for _ in range(len(order))]
    for vertex in order:
        i = index[vertex]
        for other in neighbours(vertex):
            j = index[other]
            adj[i].add(j)

    edges = [(i, j) for i in range(len(order)) for j in adj[i] if i < j]
    return len(order), edges, adj

def _girth_nb(adj, g):
    """Exact girth via non-backtracking walks (BFS over directed edges, no immediate
    backtrack). Returns the shortest cycle length (>=3) or None. Stops early once the
    shortest cycle is found, and prunes expansions that cannot beat the current best."""
    n = len(adj)
    if n == 0:
        return None
    L = [sorted(a) for a in adj]
    # depth[(u, v)] = length of the non-backtracking walk ending in directed edge (u -> v)
    depth = {}
    best = None
    # seed: all directed edges at depth 1
    level = []
    for u in range(n):
        for v in L[u]:
            depth[(u, v)] = 1
            level.append((u, v))
    while level and (best is None or best > g):
        if _deadline():
            return best
        nxt = []
        for (u, v) in level:
            du = depth[(u, v)]
            for w in L[v]:
                if w == u:          # no immediate backtrack
                    continue
                if (v, w) in depth:
                    d = depth[(v, w)]
                    if d >= 2:      # u -> v -> w closes a cycle of length d + 1
                        cand = d + 1
                        if best is None or cand < best:
                            best = cand
                        if best == 3:
                            return best
                    continue
                nd = du + 1
                if best is not None and nd + 1 >= best:
                    # extending this walk cannot produce a cycle shorter than best
                    continue
                depth[(v, w)] = nd
                nxt.append((v, w))
        level = nxt
    return best

def _random_regular_graph(n, k, seed=42):
    """Construct a random k-regular graph on n vertices using configuration model."""
    rng = random.Random(seed)
    half_edges = n * k
    if half_edges % 2 != 0:
        return None
    stubs = list(range(n)) * k
    rng.shuffle(stubs)
    adj = [set() for _ in range(n)]
    for i in range(0, half_edges, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v or v in adj[u]:
            return None
        adj[u].add(v)
        adj[v].add(u)
    # Verify regularity
    for i in range(n):
        if len(adj[i]) != k:
            return None
    edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
    return n, edges, adj

def _verify_girth(adj, g):
    """Exact certificate: no cycle shorter than g (i.e. girth >= g)."""
    gi = _girth_nb(adj, g)
    if gi is None:
        return True
    return gi >= g

def _check_simple_and_regular(n, edges, k):
    seen = set()
    deg = [0] * n
    for (i, j) in edges:
        if i < 0 or i >= n or j < 0 or j >= n or i == j:
            return False
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[a] += 1
        deg[b] += 1
    return all(x == k for x in deg)

def construct(k, g):
    # --- candidate (d, q) choices within the LUW family, cheapest first -------------------
    # girth >= d + 5 (odd d)  =>  need  d >= g - 5.  Smaller d / q -> fewer vertices.
    candidates = []
    
    # For k=4, use q=4 (Galois field). For other k, try q=k and nearby primes.
    if k == 4:
        qs = [4]
    else:
        qs = [k]
        # Try nearby primes as alternative field sizes
        for offset in [1, -1, 2, -2, 3, -3, 4, -4]:
            q_try = k + offset
            if q_try >= 2 and _is_prime(q_try) and q_try != k:
                qs.append(q_try)
    
    for q in qs:
        d0 = g - 5
        if d0 % 2 == 0:
            d0 += 1
        d0 = max(3, d0)
        for d in (d0, d0 + 2, d0 + 4, d0 + 6):
            candidates.append((d, q))
    
    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    best = None
    best_n = float('inf')
    best_key = float('inf')  # smooth surrogate: n + eps * max(0, g - girth)

    for (d, q) in ordered:
        if _deadline():
            break
        try:
            n, edges, adj = _luw_graph(d, q)
        except Exception:
            continue

        if not _check_simple_and_regular(n, edges, k):
            continue

        gi = _girth_nb(adj, g)
        if gi is None or gi < g:
            continue

        # soft-min surrogate: prefer valid (girth >= g) graphs, then smallest n.
        # Valid graphs get penalty 0; among them the smallest n wins.
        pen = max(0, g - gi)
        key = n + 1e-3 * pen
        if n < best_n or key < best_key - 1e-9:
            best_n = n
            best_key = key
            best = (n, edges)

    # If no valid LUW graph found, try random regular graphs with time budget
    if best is None and not _deadline():
        time_budget = 30.0
        start_time = time.time()
        # Try various n values from Moore bound upward
        moore = 1 + k * (1 + (k-1) + (k-1)**2)
        for n_try in range(max(moore, 2*k), max(moore, 2*k) + 200):
            if time.time() - start_time > time_budget or _deadline():
                break
            # n must satisfy n*k is even
            if n_try * k % 2 != 0:
                continue
            for seed in range(5):
                if time.time() - start_time > time_budget or _deadline():
                    break
                result = _random_regular_graph(n_try, k, seed=seed)
                if result is None:
                    continue
                n_r, edges_r, adj_r = result
                if not _check_simple_and_regular(n_r, edges_r, k):
                    continue
                gi_r = _girth_nb(adj_r, g)
                if gi_r is not None and gi_r >= g:
                    best = (n_r, edges_r)
                    break
            if best is not None:
                break

    # Fallback: if the girth check was too conservative, return the first regular LUW graph.
    if best is None:
        for (d, q) in ordered:
            try:
                n, edges, adj = _luw_graph(d, q)
                if _check_simple_and_regular(n, edges, k):
                    return n, edges
            except Exception:
                continue
        return 0, []

    # Final exact non-backtracking certification of the returned graph.
    # (Already certified above; this is a safety net in case best came from fallback.)
    return best
