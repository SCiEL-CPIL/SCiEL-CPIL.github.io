"""Combined program: LZW algebraic + greedy + bipartite random greedy."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import deque


def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Returns (n, edges).
    """
    deadline = time.time() + 112  # Safety margin (evaluator kills at 150s)

    # 1) Fast algebraic LZW construction (deterministic, valid upper bound)
    try:
        lzw = _lzw_construct(k, g)
        if lzw is not None:
            best = lzw
        else:
            best = None
    except Exception:
        best = None

    # 2) Bipartite random greedy (points/lines incidence) - the main search
    if time.time() < deadline:
        # Try a range of n values; keep the best (smallest n) valid result
        # Moore bound for k=5,g=9 is 426; best known 1116; LZW gives ~1562.
        # We search from a bit above the Moore bound upward.
        base = _moore_bound(k, g)
        candidates = []
        # Build a candidate list: start near base, grow by ~8%, cap below LZW
        cap = best[0] if best else 2000
        n = max(base + 2, 500)
        while n <= cap and time.time() < deadline:
            candidates.append(n)
            n = int(n * 1.08) + 2
        # Ensure we try a few well-spaced values
        for extra in [700, 900, 1100, 1300, 1500]:
            if extra <= cap and extra not in candidates:
                candidates.append(extra)
        candidates = sorted(set(candidates))

        for n in candidates:
            if time.time() > deadline - 5:
                break
            res = _bipartite_greedy(k, g, n, deadline)
            if res is not None:
                if best is None or res[0] < best[0]:
                    best = res
                # Found a valid one; keep searching a bit for smaller? No,
                # we go upward, so first found is smallest in this loop.
                break

    # 3) General greedy fallback (non-bipartite)
    if time.time() < deadline - 5:
        for n in [1100, 1200, 1300, 1400]:
            if time.time() > deadline - 5:
                break
            res = _greedy_construct(k, g, n, deadline)
            if res is not None:
                if best is None or res[0] < best[0]:
                    best = res
                break

    if best is not None:
        return best

    # 4) Final fallback: LZW again (should not reach here)
    return _lzw_construct(k, g)


def _moore_bound(k, g):
    """Moore bound (lower bound) for n(k,g)."""
    if g % 2 == 0:
        d = g // 2
        n = k * (k - 1) ** (d - 1)
    else:
        d = (g - 1) // 2
        n = 1 + k * sum((k - 1) ** i for i in range(d))
    return n


def _bipartite_greedy(k, g, n, deadline):
    """
    Bipartite random greedy construction.
    Vertices split into L (0..n/2-1) and R (n/2..n-1).
    Each vertex gets exactly k edges to the other side.
    Girth constraint: adding edge (u in L, v in R) creates a cycle of length
    2*(dist(u,v)+1) where dist is in the current bipartite graph.
    We require 2*(dist+1) >= g, i.e., dist >= ceil(g/2)-1.
    For g=9: dist >= 4.
    """
    n_half = n // 2
    if 2 * n_half != n:
        n = 2 * n_half

    # Cycle length = 2*(dist+1). Need 2*(dist+1) >= g => dist >= ceil(g/2) - 1.
    # For g=9: ceil(4.5)=5, 5-1=4.
    min_dist = -(-g // 2) - 1

    max_restarts = 10
    for _ in range(max_restarts):
        if time.time() > deadline - 3:
            return None

        adj = [set() for _ in range(n)]
        deg = [0] * n

        # Process vertices in random order, but try to balance sides
        order = list(range(n))
        random.shuffle(order)

        success = True
        for u in order:
            if deg[u] >= k:
                continue
            side = 0 if u < n_half else 1
            # Find all valid v on the other side
            valid_vs = []
            other_range = range(n_half, n) if side == 0 else range(n_half)
            for v in other_range:
                if deg[v] >= k:
                    continue
                if v in adj[u]:
                    continue
                # Check distance between u and v in current graph
                d = _bipartite_distance(adj, u, v, min_dist)
                if d >= min_dist:
                    valid_vs.append(v)

            if not valid_vs:
                success = False
                break

            # Pick from lower-degree half to promote balance
            valid_vs.sort(key=lambda v: deg[v])
            half = len(valid_vs) // 2
            pool = valid_vs[:max(1, half)]
            v = random.choice(pool)

            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1

        if not success:
            continue

        # Check all degrees are k
        if any(d != k for d in deg):
            continue

        # Build edge list and verify girth
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))

        if _verify_girth_fast(n, edges, k, g):
            return (n, edges)

    return None


def _bipartite_distance(adj, u, v, limit):
    """
    BFS distance from u to v in the current bipartite graph.
    Returns distance if <= limit, else limit+1 (meaning "far enough").
    """
    if u == v:
        return 0
    dist = {u: 0}
    queue = deque([u])
    while queue:
        node = queue.popleft()
        d = dist[node]
        if d >= limit:
            continue
        for nb in adj[node]:
            if nb == v:
                return d + 1
            if nb not in dist:
                dist[nb] = d + 1
                queue.append(nb)
    return limit + 1


def _verify_girth_fast(n, edges, k, g):
    """
    Verify k-regular and girth >= g using BFS from each vertex.
    Uses numpy for speed.
    """
    # Check degree and duplicates via edge count
    if len(edges) != n * k // 2:
        return False
    edge_set = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)

    # Build adjacency as list of arrays
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        if len(adj[i]) != k:
            return False

    # Convert to numpy arrays for faster BFS
    adj_np = [np.array(a, dtype=np.int32) for a in adj]

    # BFS from each vertex to find shortest cycle
    # We only need to check up to depth (g-1)//2 from each start
    max_depth = (g - 1) // 2  # For g=9, max_depth=4; cycles of length < 9 have a "midpoint" at depth <= 4

    for start in range(n):
        dist = np.full(n, -1, dtype=np.int32)
        parent = np.full(n, -1, dtype=np.int32)
        dist[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            d = dist[node]
            if d >= max_depth:
                continue
            for nb in adj_np[node]:
                nb = int(nb)
                if dist[nb] == -1:
                    dist[nb] = d + 1
                    parent[nb] = node
                    queue.append(nb)
                elif nb != parent[node]:
                    cycle_len = int(dist[node]) + int(dist[nb]) + 1
                    if cycle_len < g:
                        return False
    return True


def _greedy_construct(k, g, n, deadline):
    """
    General (non-bipartite) greedy construction.
    """
    max_restarts = 3
    for _ in range(max_restarts):
        if time.time() > deadline - 3:
            return None

        adj = [set() for _ in range(n)]
        deg = [0] * n

        order = list(range(n))
        random.shuffle(order)

        success = True
        for u in order:
            if deg[u] >= k:
                continue
            valid_vs = []
            for v in range(n):
                if v == u or v in adj[u] or deg[v] >= k:
                    continue
                # Check girth constraint: adding (u,v) must not create cycle < g
                # BFS from u to v; if dist < g-1, adding creates cycle of length dist+1 < g
                # Actually: if there's a path of length L from u to v, adding edge gives cycle L+1.
                # Need L+1 >= g => L >= g-1.
                d = _general_distance(adj, u, v, g - 1)
                if d >= g - 1:
                    valid_vs.append(v)

            if not valid_vs:
                success = False
                break

            valid_vs.sort(key=lambda v: deg[v])
            half = len(valid_vs) // 2
            pool = valid_vs[:max(1, half)]
            v = random.choice(pool)

            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1

        if not success:
            continue

        if any(d != k for d in deg):
            continue

        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))

        if _verify_girth_fast(n, edges, k, g):
            return (n, edges)

    return None


def _general_distance(adj, u, v, limit):
    """BFS distance from u to v, capped at limit."""
    if u == v:
        return 0
    dist = {u: 0}
    queue = deque([u])
    while queue:
        node = queue.popleft()
        d = dist[node]
        if d >= limit:
            continue
        for nb in adj[node]:
            if nb == v:
                return d + 1
            if nb not in dist:
                dist[nb] = d + 1
                queue.append(nb)
    return limit + 1


def _lzw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar graph"""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
# EVOLVE-BLOCK-END
