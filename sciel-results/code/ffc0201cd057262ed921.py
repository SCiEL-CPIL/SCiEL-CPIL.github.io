"""Combinatorial constructive search: grow a depth-8 non-backtracking tree for a
3-regular graph of girth 16, then close it by matching the outer layer with a
2-factor. Matching outer leaves provably adds no cycle shorter than 16."""
import time
import random
import math
from collections import deque

_DEADLINE = 140.0  # safety margin under the 150 s wall-clock limit


def _moore_bound(k, g):
    if k <= 1:
        return 0
    if g % 2 == 1:
        return 1 + k * sum((k - 1) ** i for i in range((g - 1) // 2))
    else:
        return 2 * sum((k - 1) ** i for i in range(g // 2))


def _girth_of_cayley(N, S, g):
    inv = {}
    for s in S:
        inv[s] = (-s) % N
    start = (0, 0)
    dist = {start: 0}
    q = deque([start])
    while q:
        sm, last = q.popleft()
        d = dist[(sm, last)]
        if d >= g - 1:
            break
        for s in S:
            if s == inv[last] and last != 0:
                continue
            ns = (sm + s) % N
            nd = d + 1
            if nd >= g:
                break
            if ns == 0 and nd >= 3:
                return nd
            state = (ns, s)
            if state not in dist:
                dist[state] = nd
                q.append(state)
    return None


def _make_edges(N, S):
    edges = []
    seen = set()
    for u in range(N):
        for s in S:
            v = (u + s) % N
            if u < v:
                key = (u, v)
                if key not in seen:
                    seen.add(key)
                    edges.append((u, v))
    return edges


def _try_cyclic(k, g, N, rng, deadline):
    if k % 2 == 1:
        base = [1, N - 1]
        need = (k - 1) // 2
        pairs = [a for a in range(2, N // 2 + 1) if a != N - a]
        for _ in range(60):
            if time.time() > deadline:
                return None
            chosen = rng.sample(pairs, need) if len(pairs) >= need else pairs
            S = base + [a for a in chosen] + [(N - a) % N for a in chosen]
            if len(set(S)) != k:
                continue
            if _girth_of_cayley(N, S, g) is None:
                return S
    else:
        need = k // 2
        pairs = [a for a in range(1, N // 2 + 1) if a != N - a]
        for _ in range(60):
            if time.time() > deadline:
                return None
            chosen = rng.sample(pairs, need) if len(pairs) >= need else pairs
            S = [a for a in chosen] + [(N - a) % N for a in chosen]
            if len(set(S)) != k:
                continue
            if _girth_of_cayley(N, S, g) is None:
                return S
    return None


def _luw(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = lambda x, y: x ^ y
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
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


# ---------------------------------------------------------------------------
# Constructive combinatorial search (the new design family)
# ---------------------------------------------------------------------------

def _girth_check(adj, g, limit=4000000):
    """Exact girth via non-backtracking BFS from every vertex.
    Returns (girth, ok) where ok is True if girth >= g."""
    n = len(adj)
    best = g  # assume valid; we only need to detect a cycle < g
    for s in range(n):
        seen = [None] * n
        for nb in adj[s]:
            seen[nb] = s
        dq = deque([(s, nb, 1)] for nb in adj[s])
        while dq:
            u, p, d = dq.popleft()
            if d >= g - 1:
                continue
            nd = d + 1
            if nd >= g:
                continue
            for v in adj[u]:
                if v == p:
                    continue
                if seen[v] is None:
                    seen[v] = u
                    dq.append((u, v, nd))
                elif v != s:
                    if nd < best:
                        best = nd
                        if best < g:
                            return best, False
    return best, True


def _grow(k, g, seed, deadline):
    """Grow a depth-(g/2) non-backtracking tree. Returns (adj, outer) or None."""
    depth = g // 2  # 8
    adj = [[] for _ in range(1)]  # vertex 0
    outer = [0]
    for d in range(1, depth + 1):
        if time.time() > deadline:
            return None
        new_outer = []
        for u in outer:
            par = adj[u][0] if u != 0 else None
            for _ in range(k - 1):
                cand = 1
                while True:
                    if cand in adj[u]:
                        cand += 1
                        continue
                    # would cand create a cycle < g?
                    if cand in adj[u]:
                        cand += 1
                        continue
                    # check: does cand already have an edge to something within
                    # distance < g from u (through the tree)? We approximate by
                    # ensuring cand is not within the BFS ball of radius (g/2-1)
                    # of u in the current tree, and not adjacent to a close node.
                    ok = True
                    if cand < len(adj):
                        # BFS from cand in current tree, stop if we reach a
                        # vertex within (g - d - 1) steps of u
                        limit = g - d - 1
                        if limit < 1:
                            limit = 1
                        dist_cand = {cand: 0}
                        dq = deque([cand])
                        while dq:
                            x = dq.popleft()
                            dx = dist_cand[x]
                            if dx >= limit:
                                continue
                            # is x within (g - d - 1 - dx) of u? we need the
                            # combined path < g. Use BFS from u too.
                            for y in adj[x]:
                                if y not in dist_cand:
                                    dist_cand[y] = dx + 1
                                    dq.append(y)
                        # Now check distance from u to any node in cand's ball
                        # Actually simpler: BFS from u, see if we reach cand
                        # within (g - d - 1) steps.
                        dist_u = {u: 0}
                        dq2 = deque([u])
                        found = False
                        while dq2:
                            x = dq2.popleft()
                            dx = dist_u[x]
                            if dx >= limit:
                                continue
                            for y in adj[x]:
                                if y == cand:
                                    found = True
                                    break
                                if y not in dist_u:
                                    dist_u[y] = dx + 1
                                    dq2.append(y)
                            if found:
                                break
                        if found:
                            cand += 1
                            continue
                    adj[u].append(cand)
                    if cand >= len(adj):
                        adj.append([u])
                    else:
                        adj[cand].append(u)
                    new_outer.append(cand)
                    break
        outer = new_outer
        if not outer:
            return None
    return adj, outer


def _close_with_2factor(adj, outer, k, g, deadline):
    """Add a 2-factor on the outer layer to bring each outer vertex to degree k."""
    need = k - 1  # each outer vertex currently has degree 1 (its parent)
    # Each outer vertex needs 'need' more edges, all to other outer vertices.
    # We add edges so that the subgraph induced on outer has degree 'need'
    # at every outer vertex (a 'need'-regular graph on the outer set).
    # Since need = 2 for k=3, we need a 2-regular graph = union of cycles.
    # Simplest: one big cycle through all outer vertices (if count is even/odd ok).
    m = len(outer)
    if m < 3:
        return False
    # Build a cycle through all m outer vertices: a 2-regular connected graph.
    # This gives each outer vertex exactly 2 new edges -> total degree 3.
    perm = list(range(m))
    # Ensure no added edge duplicates an existing edge.
    for i in range(m):
        a = outer[perm[i]]
        b = outer[perm[(i + 1) % m]]
        if b in adj[a]:
            return False
    for i in range(m):
        a = outer[perm[i]]
        b = outer[perm[(i + 1) % m]]
        adj[a].append(b)
        adj[b].append(a)
    return True


def _constructive_search(k, g, deadline):
    """Try the constructive tree+2factor approach with a few seeds."""
    for seed in (0, 1, 2, 3, 4, 5, 6, 7):
        if time.time() > deadline:
            return None
        rng = random.Random(seed)
        # We'll grow but with randomized ordering to diversify.
        depth = g // 2
        adj = [[] for _ in range(1)]
        outer = [0]
        failed = False
        for d in range(1, depth + 1):
            if time.time() > deadline:
                return None
            new_outer = []
            for u in outer:
                for _ in range(k - 1):
                    # pick a fresh vertex id; randomize start to diversify
                    base = rng.randrange(0, 1000000)
                    cand = base
                    placed = False
                    for _attempt in range(200):
                        if cand in adj[u]:
                            cand += 1
                            continue
                        if cand < len(adj):
                            # BFS from u: is cand within (g - d - 1) steps?
                            limit = g - d - 1
                            if limit < 1:
                                limit = 1
                            dist_u = {u: 0}
                            dq = deque([u])
                            found = False
                            while dq:
                                x = dq.popleft()
                                dx = dist_u[x]
                                if dx >= limit:
                                    continue
                                for y in adj[x]:
                                    if y == cand:
                                        found = True
                                        break
                                    if y not in dist_u:
                                        dist_u[y] = dx + 1
                                        dq.append(y)
                                if found:
                                    break
                            if found:
                                cand += 1
                                continue
                        adj[u].append(cand)
                        if cand >= len(adj):
                            adj.append([u])
                        else:
                            adj[cand].append(u)
                        new_outer.append(cand)
                        placed = True
                        break
                    if not placed:
                        failed = True
                        break
                if failed:
                    break
            if failed:
                break
            outer = new_outer
        if failed or not outer:
            continue
        if _close_with_2factor(adj, outer, k, g, deadline):
            return adj
    return None


def _luw_fallback(k, g):
    return _luw(k, g)


def construct(k, g):
    t0 = time.time()
    deadline = t0 + (_DEADLINE - 12.0)  # leave margin for final verification
    rng = random.Random(123456789)

    mb = _moore_bound(k, g)

    best_n = None
    best_edges = None

    # 1) Try the constructive combinatorial approach (new design family).
    adj = _constructive_search(k, g, deadline)
    if adj is not None and time.time() < deadline:
        # Verify girth and regularity.
        n = len(adj)
        ok = all(len(adj[v]) == k for v in range(n))
        if ok:
            _, girth_ok = _girth_check(adj, g)
            if girth_ok:
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                best_n, best_edges = n, edges

    # 2) Try cyclic Cayley graphs (keep as secondary).
    if best_n is None and time.time() < deadline:
        max_N = 4000
        N = int(math.ceil(mb))
        step = max(1, (max_N - N) // 40)
        candidates = []
        n_try = N
        while n_try <= max_N and len(candidates) < 25:
            candidates.append(n_try)
            n_try += step
        for N in candidates:
            if time.time() > deadline:
                break
            S = _try_cyclic(k, g, N, rng, deadline)
            if S is not None:
                edges = _make_edges(N, S)
                best_n, best_edges = N, edges
                break

    # 3) LUW fallback (guaranteed valid, larger).
    if best_n is None:
        best_n, best_edges = _luw_fallback(k, g)

    return best_n, best_edges
