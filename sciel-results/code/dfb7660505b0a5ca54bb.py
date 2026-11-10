"""Improved n(5,9) cage construction: enumerate explicit algebraic families,
then polish k-regularity-preserving edge swaps under a girth constraint.
Keeps the construct(k, g) -> (n, edges) interface."""
import time
import random
from collections import deque

_DEADLINE = 115.0  # safety margin vs 120 s wall-clock


def _bipartite_cayley(q, coeffs):
    """Bipartite Cayley graph over GF(q) with adjacency via quadratic form.
    Left i ~ right j iff sum_a coeffs[a]*(a*i + j) == 0 in GF(q).
    Returns (n, edges) for the connected component of (0,0)."""
    if q % 2 == 0:
        return None
    adj = {}
    for i in range(q):
        for j in range(q):
            s = 0
            for a, c in coeffs.items():
                s = (s + c * ((a * i + j) % q)) % q
            if s == 0:
                adj[(i, j)] = True
    # degree check
    deg = 0
    for a, c in coeffs.items():
        pass
    # count degree of (0,0): j such that sum c*a*j == 0
    deg = sum(1 for j in range(q) if sum(c * a * j for a, c in coeffs.items()) % q == 0)
    # Build adjacency lists
    left = [list(range(q))]
    right = [list(range(q))]
    for i in range(q):
        for j in range(q):
            s = 0
            for a, c in coeffs.items():
                s = (s + c * ((a * i + j) % q)) % q
            if s == 0:
                left[i].append(q + j)
                right[q + j].append(i)
    # connected component of 0
    start = 0
    seen = {start}
    dq = deque([start])
    comp = []
    while dq:
        v = dq.popleft()
        comp.append(v)
        for w in left[v] if v < q else right[v]:
            if w not in seen:
                seen.add(w)
                dq.append(w)
    n = len(comp)
    if n != 2 * q:
        return None  # not connected, skip
    idx = {v: i for i, v in enumerate(comp)}
    edges = []
    for v in comp:
        if v < q:
            for w in left[v]:
                if idx[v] < idx[w]:
                    edges.append((idx[v], idx[w]))
    return n, edges


def _generalized_petersen(n, k):
    """Generalized Petersen G(n,k): 2n vertices, 3-regular.
    Outer cycle i~i+1, inner i~i+k, spokes i~n+i."""
    if n < 5:
        return None
    edges = []
    for i in range(n):
        edges.append((i, (i + 1) % n))
        edges.append((n + i, n + (i + k) % n))
        edges.append((i, n + i))
    # dedup
    edges = list(set(edges))
    return 2 * n, edges


def _verify(k, g, n, edges):
    """Return True if graph is simple, k-regular, and girth >= g."""
    if not (0 <= n):
        return False
    deg = [0] * n
    seen_edges = set()
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in seen_edges:
            return False
        seen_edges.add(key)
        deg[u] += 1
        deg[v] += 1
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        if deg[i] != k:
            return False
    # girth via non-backtracking BFS from each vertex
    for s in range(n):
        dist = {s: 0}
        # non-backtracking: state (vertex, prev)
        dq = deque()
        for nb in adj[s]:
            dq.append((nb, s, 1))
            dist[(nb, s)] = 1
        while dq:
            v, prev, d = dq.popleft()
            if d >= g:
                break
            for nb in adj[v]:
                if nb == prev:
                    continue
                if (nb, v) in dist:
                    # found a cycle of length d + dist[(nb, v)]
                    cyc = d + 1 + dist[(nb, v)]
                    if cyc < g:
                        return False
                    continue
                dist[(nb, v)] = d + 1
                dq.append((nb, v, d + 1))
    return True


def _short_walk_count(adj, n, g):
    """Surrogate objective: number of non-backtracking walks of length < g/2."""
    half = g // 2
    total = 0
    for s in range(n):
        cnt = {s: 1}
        dq = deque([(s, None, 0)])
        while dq:
            v, prev, d = dq.popleft()
            if d >= half:
                continue
            for nb in adj[v]:
                if nb == prev:
                    continue
                cnt[nb] = cnt.get(nb, 0) + 1
                dq.append((nb, v, d + 1))
        total += sum(cnt.values())
    return total


def construct(k, g):
    t0 = time.time()
    rng = random.Random(12345)

    # ---- Phase 1: enumerate explicit families, keep smallest valid ----
    best = None  # (n, edges, adj)
    candidates = []

    # Bipartite Cayley over GF(q) for odd q
    for q in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        if 2 * q < k * 2:  # trivial: need at least k vertices per side
            pass
        # degree of (0,0) for coeffs {a: c} is count of j with sum c*a*j == 0
        # For a single coefficient a=1, c=1: j such that j == 0 -> degree 1. Not enough.
        # Use multiple a's to get degree k.
        # Try small coefficient sets
        for ncoeff in [1, 2, 3]:
            for a1 in range(1, q):
                for a2 in range(1, q) if ncoeff >= 2 else [0]:
                    for a3 in range(1, q) if ncoeff >= 3 else [0]:
                        coeffs = {a1: 1}
                        if ncoeff >= 2:
                            coeffs[a2] = 1
                        if ncoeff >= 3:
                            coeffs[a3] = 1
                        # degree: count j with sum a_i * j == 0
                        deg = sum(1 for j in range(q)
                                  if sum(a * j for a in coeffs) % q == 0)
                        if deg == k:
                            res = _bipartite_cayley(q, coeffs)
                            if res is not None:
                                candidates.append(res)
        if time.time() - t0 > 5:
            break

    # Generalized Petersen (only useful for small k, but include for completeness)
    if k == 3:
        for n in range(5, 200):
            res = _generalized_petersen(n, 1)
            if res is not None:
                candidates.append(res)

    # Pick the smallest valid candidate
    for n, edges in sorted(candidates, key=lambda x: x[0]):
        if _verify(k, g, n, edges):
            adj = [set() for _ in range(n)]
            for (u, v) in edges:
                adj[u].add(v)
                adj[v].add(u)
            best = (n, edges, adj)
            break

    if best is None:
        # Fallback: use the seed's L-U-W construction (reimplemented compactly)
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
        n = len(order)
        adj = [set() for _ in range(n)]
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
        best = (n, edges, adj)

    n, edges, adj = best

    # ---- Phase 2: local edge-swap polish ----
    # Replace edges (u,v) with (u,w),(v,z) preserving k-regularity and girth.
    # Objective: reduce short-walk count.
    edge_set = set((min(u, v), max(u, v)) for (u, v) in edges)

    def try_swap(u, v, w, z):
        """Replace edge (u,v) with edges (u,w) and (v,z)."""
        if u == w or v == z or u == z or v == w:
            return False
        if (min(u, w), max(u, w)) in edge_set:
            return False
        if (min(v, z), max(v, z)) in edge_set:
            return False
        # remove (u,v), add (u,w),(v,z)
        edge_set.discard((min(u, v), max(u, v)))
        edge_set.add((min(u, w), max(u, w)))
        edge_set.add((min(v, z), max(v, z)))
        adj[u].discard(v)
        adj[v].discard(u)
        adj[u].add(w)
        adj[v].add(z)
        adj[w].add(u)
        adj[z].add(v)
        if not _verify(k, g, n, list(edge_set)):
            # revert
            edge_set.discard((min(u, w), max(u, w)))
            edge_set.discard((min(v, z), max(v, z)))
            edge_set.add((min(u, v), max(u, v)))
            adj[u].discard(w)
            adj[v].discard(z)
            adj[u].add(v)
            adj[v].add(u)
            adj[w].discard(u)
            adj[z].discard(v)
            return False
        return True

    # Simple hill-climbing: try random swaps that reduce short-walk count
    if time.time() - t0 < 100:
        current_obj = _short_walk_count(adj, n, g)
        for _ in range(20000):
            if time.time() - t0 > 105:
                break
            u, v = rng.sample(range(n), 2)
            if v not in adj[u]:
                continue
            w = rng.choice(list(adj[u]))
            z = rng.choice(list(adj[v]))
            if w == v or z == u:
                continue
            if (min(u, w), max(u, w)) in edge_set or (min(v, z), max(v, z)) in edge_set:
                continue
            # attempt swap
            old = (min(u, v), max(u, v))
            edge_set.discard(old)
            edge_set.add((min(u, w), max(u, w)))
            edge_set.add((min(v, z), max(v, z)))
            adj[u].discard(v)
            adj[v].discard(u)
            adj[u].add(w)
            adj[v].add(z)
            adj[w].add(u)
            adj[z].add(v)
            if _verify(k, g, n, list(edge_set)):
                new_obj = _short_walk_count(adj, n, g)
                if new_obj < current_obj:
                    current_obj = new_obj
                else:
                    # revert
                    edge_set.discard((min(u, w), max(u, w)))
                    edge_set.discard((min(v, z), max(v, z)))
                    edge_set.add(old)
                    adj[u].discard(w)
                    adj[v].discard(z)
                    adj[u].add(v)
                    adj[v].add(u)
                    adj[w].discard(u)
                    adj[z].discard(v)
            else:
                # revert
                edge_set.discard((min(u, w), max(u, w)))
                edge_set.discard((min(v, z), max(v, z)))
                edge_set.add(old)
                adj[u].discard(w)
                adj[v].discard(z)
                adj[u].add(v)
                adj[v].add(u)
                adj[w].discard(u)
                adj[z].discard(v)

    final_edges = sorted(edge_set)
    return n, final_edges
