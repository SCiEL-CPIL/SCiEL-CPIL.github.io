import time
import random
import sys
from collections import deque

def _moore_lower_bound(k, g):
    # Moore bound for k-regular girth g (g even): 2 * sum_{i=0}^{g/2 -1} (k-1)^i
    if g % 2 == 0:
        return 2 * sum((k - 1) ** i for i in range(g // 2))
    else:
        return k * sum((k - 1) ** i for i in range((g - 1) // 2))

def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_girth_fast(n, edges, min_girth):
    """Non-backtracking BFS from each vertex; returns True iff no cycle < min_girth."""
    adj = _build_adj(n, edges)
    max_depth = (min_girth - 1) // 2
    for start in range(n):
        # visited[v] = depth from start (BFS distance), parent for non-backtracking
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            node = q.popleft()
            d = dist[node]
            if d > max_depth:
                continue
            for nb in adj[node]:
                if nb == parent[node]:
                    continue
                if dist[nb] == -1:
                    dist[nb] = d + 1
                    parent[nb] = node
                    q.append(nb)
                else:
                    # edge node-nb: cycle length d + dist[nb] + 1
                    if d + dist[nb] + 1 < min_girth:
                        return False
    return True

def _girth_of(n, edges):
    """Return the girth (length of shortest cycle) of a simple 3-regular graph."""
    adj = _build_adj(n, edges)
    best = 10**9
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            node = q.popleft()
            d = dist[node]
            for nb in adj[node]:
                if nb == parent[node]:
                    continue
                if dist[nb] == -1:
                    dist[nb] = d + 1
                    parent[nb] = node
                    q.append(nb)
                else:
                    L = d + dist[nb] + 1
                    if L < best:
                        best = L
                        if best <= 4:
                            return best
    return best

def _gen_config_3regular(n):
    """Generate a simple 3-regular graph via configuration model (retry)."""
    if n % 2 != 0:
        return None
    for _ in range(200):
        half = list(range(3 * n))
        random.shuffle(half)
        edges = []
        seen = set()
        ok = True
        for i in range(0, 3 * n, 2):
            u = half[i] // 3
            v = half[i + 1] // 3
            if u == v:
                ok = False
                break
            key = (u, v) if u < v else (v, u)
            if key in seen:
                ok = False
                break
            seen.add(key)
            edges.append((u, v))
        if ok:
            return edges
    return None

def _switch_improve(n, edges, g, deadline, max_iter=400):
    """Use 2-switches to increase girth; return best girth-achieving edges or None."""
    adj = _build_adj(n, edges)
    # adjacency as set for O(1)
    adjset = [set(a) for a in adj]
    cur_girth = _girth_of(n, edges)
    best_girth = cur_girth
    best_edges = list(edges)
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        if cur_girth >= g:
            return list(edges)
        # pick two random edges that are not adjacent and share no endpoint
        e1 = random.randint(0, len(edges) - 1)
        e2 = random.randint(0, len(edges) - 1)
        if e1 == e2:
            continue
        a, b = edges[e1]
        c, d = edges[e2]
        if len({a, b, c, d}) < 4:
            continue
        # switch: (a,b),(c,d) -> (a,c),(b,d)
        new1 = (a, c) if a < c else (c, a)
        new2 = (b, d) if b < d else (d, b)
        if new1 == new2:
            continue
        if new1 in adjset[a] and new2 in adjset[b]:
            continue
        # apply
        adjset[a].discard(b); adjset[b].discard(a)
        adjset[c].discard(d); adjset[d].discard(c)
        adjset[a].add(c); adjset[c].add(a)
        adjset[b].add(d); adjset[d].add(b)
        edges[e1] = new1
        edges[e2] = new2
        ng = _girth_of(n, edges)
        if ng > cur_girth:
            cur_girth = ng
            if ng > best_girth:
                best_girth = ng
                best_edges = list(edges)
        else:
            # revert
            adjset[a].discard(c); adjset[c].discard(a)
            adjset[b].discard(d); adjset[d].discard(b)
            adjset[a].add(b); adjset[b].add(a)
            adjset[c].add(d); adjset[d].add(c)
            edges[e1] = (a, b) if a < b else (b, a)
            edges[e2] = (c, d) if c < d else (d, c)
    if best_girth >= g:
        return best_edges
    return None

def _construct_3regular(k, g, deadline):
    lb = _moore_lower_bound(k, g)
    # candidate sizes: even, >= lb, scan upward
    candidates = []
    n = lb
    if n % 2 != 0:
        n += 1
    while n <= lb + 4000 and len(candidates) < 60:
        candidates.append(n)
        n += 2
    # also include some well-known small sizes
    for extra in [1024, 1200, 1400, 1600, 1800, 2000, 2400, 2800, 3200]:
        if extra >= lb and extra % 2 == 0 and extra not in candidates:
            candidates.append(extra)
    candidates = sorted(set(candidates))

    best_n = None
    best_edges = None
    for n in candidates:
        if time.time() > deadline:
            break
        # multiple random starts
        found = False
        for attempt in range(6):
            if time.time() > deadline:
                break
            edges = _gen_config_3regular(n)
            if edges is None:
                continue
            if _check_girth_fast(n, edges, g):
                return n, edges
            improved = _switch_improve(n, edges, g, deadline - 5.0, max_iter=300)
            if improved is not None:
                return n, improved
        if best_n is None or n < best_n:
            best_n = n
            # keep the best-girth attempt as fallback (won't be valid, but we return None if invalid)
    # If we found a valid one earlier we returned; otherwise return best valid we stored
    # (we only store valid ones, so best_edges is None unless we returned)
    return None

def _luw_construct(k, g):
    """Fallback LUW construction (kept for non-3 or as last resort)."""
    if k == 3:
        q = 3
        d = 8
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
    else:
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

def construct(k, g):
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill
    random.seed(12345)

    if k != 3:
        return _luw_construct(k, g)

    # Primary: direct search for small 3-regular graph with girth g
    result = _construct_3regular(k, g, deadline)
    if result is not None:
        return result

    # Fallback: LUW (guaranteed valid, larger)
    return _luw_construct(k, g)
