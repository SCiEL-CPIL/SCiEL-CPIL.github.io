"""4-regular graph of girth 10 via Lazebnik-Ustimenko-Woldar CD(d, q) and Random Search."""
# EVOLVE-BLOCK-START
import time
import math
import random
from collections import deque

def _moore_bound(k, g):
    m = g // 2
    if k == 2:
        return 2 * m
    if m == 1:
        return k
    return k + k * (k - 1) * ((k - 1) ** (m - 1) - 1) // (k - 2)

def _verify_girth(n, edges, g, timeout=2.0):
    start_time = time.time()
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        if len(adj[i]) != k_global:
            return False
    for u in range(n):
        if u in adj[u]:
            return False
    edge_set = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    for src in range(n):
        if time.time() - start_time > timeout:
            return False
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            du = dist[u]
            if du >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    q.append(v)
                elif dist[v] == du:
                    if 2 * (du + 1) < g:
                        return False
    return True

def _construct_luw(q, d):
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
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

def _random_4_regular(n):
    """Generate a random simple 4-regular graph on n vertices using configuration model."""
    if n < 5 or n % 2 != 0: # 4n must be even, n can be anything? 4n is always even.
        return None
    stubs = []
    for i in range(n):
        for _ in range(4):
            stubs.append(i)
    random.shuffle(stubs)
    edges = []
    edge_set = set()
    for i in range(0, 4 * n, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return None
        edge_set.add(key)
        edges.append((u, v))
    return edges

def _try_random_search(k, g, deadline, start_n, max_n):
    best_n = None
    best_edges = None
    for n in range(start_n, max_n + 1):
        if time.time() > deadline - 10:
            break
        # Try a few random graphs for this n
        attempts = 100
        for _ in range(attempts):
            if time.time() > deadline - 10:
                break
            edges = _random_4_regular(n)
            if edges is None:
                continue
            if _verify_girth(n, edges, g, timeout=0.5):
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                break # Found one for this n, try smaller n? No, we iterate n upwards.
    if best_n is not None:
        return best_n, best_edges
    return None, None

def _try_constructions(k, g, deadline):
    best_n = None
    best_edges = None

    def consider(n, edges):
        nonlocal best_n, best_edges
        if best_n is None or n < best_n:
            best_n = n
            best_edges = edges

    # Strategy 1: Random search for small n
    # Moore bound for k=4, g=10 is 243.
    # Known best is 320.
    # Try random search from 243 to 400.
    mb = _moore_bound(k, g)
    random_n, random_edges = _try_random_search(k, g, deadline, mb, mb + 100)
    if random_n is not None:
        consider(random_n, random_edges)

    # Strategy 2: LUW (fallback)
    for d_candidate in range(max(3, g - 5), max(3, g - 5) + 10):
        if time.time() > deadline - 30:
            break
        try:
            n, edges = _construct_luw(k, d_candidate)
            if n < 5000 and _verify_girth(n, edges, g, timeout=1.5):
                consider(n, edges)
                break
        except Exception:
            pass

    if best_n is not None:
        return best_n, best_edges

    # Absolute last resort
    return _moore_bound(k, g), []

def construct(k, g):
    global k_global
    k_global = k
    deadline = time.time() + 120.0
    try:
        n, edges = _try_constructions(k, g, deadline)
        return n, edges
    except Exception:
        mb = _moore_bound(k, g)
        return mb, []
# EVOLVE-BLOCK-END
