"""Decompose three fixed graphs into cycles and edges, minimizing piece count."""
import time
import random
from collections import defaultdict, deque

_DEADLINE_SECONDS = 145.0


def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _bfs_path(adj, s, t):
    """Return list of vertices on a shortest path s..t, or None."""
    if s == t:
        return [s]
    parent = {s: None}
    q = deque([s])
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in parent:
                parent[v] = u
                if v == t:
                    # reconstruct
                    path = [v]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    path.reverse()
                    return path
                q.append(v)
    return None


def _longest_cycle(adj, start, n, remaining, deadline):
    if len(adj.get(start, ())) < 2:
        return None
    nbrs = {}
    for u in range(n):
        a = adj.get(u)
        if not a:
            continue
        lst = [v for v in sorted(a) if v >= start and (min(u, v), max(u, v)) in remaining]
        nbrs[u] = lst
    if len(nbrs.get(start, [])) < 2:
        return None
    best = None
    best_len = 0
    path = [start]
    on_path = {start}
    stack = [(start, 0)]
    while stack:
        if time.time() > deadline:
            break
        node, idx = stack[-1]
        if idx >= len(nbrs.get(node, [])):
            stack.pop()
            if path and path[-1] == node:
                path.pop()
            on_path.discard(node)
            continue
        v = nbrs[node][idx]
        stack[-1] = (node, idx + 1)
        if v == start:
            if len(path) >= 3 and len(path) > best_len:
                best_len = len(path)
                best = path[:]
            continue
        if v not in on_path:
            on_path.add(v)
            path.append(v)
            stack.append((v, 0))
    return best


def _eulerian_cycles(adj, n, remaining, deadline):
    """Extract cycles from an all-even graph until empty. Returns list of cycles."""
    cycles = []
    while remaining:
        if time.time() > deadline:
            break
        best_cycle = None
        best_len = 0
        for start in range(n):
            if time.time() > deadline:
                break
            if len(adj.get(start, ())) < 2:
                continue
            c = _longest_cycle(adj, start, n, remaining, deadline)
            if c and len(c) > best_len:
                best_len = len(c)
                best_cycle = c
                if best_len >= n:
                    break
        if best_cycle:
            valid = True
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                if (min(u, v), max(u, v)) not in remaining:
                    valid = False
                    break
            if valid:
                cycles.append(best_cycle)
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    e = (min(u, v), max(u, v))
                    remaining.discard(e)
                    adj[u].discard(v)
                    adj[v].discard(u)
                continue
        # should not happen in Eulerian graph, but safety: take an edge's cycle
        # find any cycle via DFS from an edge
        e = next(iter(remaining))
        u0, v0 = e
        # find a cycle containing this edge
        cyc = None
        for s in sorted({u0, v0}):
            c = _longest_cycle(adj, s, n, remaining, deadline)
            if c:
                cyc = c
                break
        if cyc:
            cycles.append(cyc)
            for i in range(len(cyc)):
                a = cyc[i]
                b = cyc[(i + 1) % len(cyc)]
                ee = (min(a, b), max(a, b))
                remaining.discard(ee)
                adj[a].discard(b)
                adj[b].discard(a)
        else:
            # truly no cycle: single edge (shouldn't happen)
            cycles.append([u0, v0])
            remaining.discard(e)
            adj[u0].discard(v0)
            adj[v0].discard(u0)
    return cycles


def _decompose(edges, n, deadline):
    if not edges:
        return []
    edges_set = set((min(u, v), max(u, v)) for u, v in edges)
    adj = defaultdict(set)
    deg = defaultdict(int)
    for u, v in edges_set:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1

    T = [v for v in range(n) if deg.get(v, 0) % 2 == 1]

    # build distance for T-join matching
    # all-pairs shortest among T via BFS
    dist = {}
    for s in T:
        d = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj.get(u, ()):
                if v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        dist[s] = d

    # minimum weight perfect matching on T (metric closure)
    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        m = len(T)
        D = np.full((m, m), 10**9)
        for i in range(m):
            for j in range(m):
                if i != j and T[j] in dist[T[i]]:
                    D[i, j] = dist[T[i]][T[j]]
        row, col = linear_sum_assignment(D)
        pairs = []
        seen = set()
        for i, j in zip(row, col):
            if i != j:
                key = (min(i, j), max(i, j))
                if key not in seen:
                    seen.add(key)
                    pairs.append((T[i], T[j]))
    except Exception:
        pairs = []
        used = [False] * len(T)
        for i in range(len(T)):
            if used[i]:
                continue
            bestj = -1
            bestd = 10**9
            for j in range(i + 1, len(T)):
                if not used[j] and T[j] in dist[T[i]]:
                    dd = dist[T[i]][T[j]]
                    if dd < bestd:
                        bestd = dd
                        bestj = j
            if bestj >= 0:
                pairs.append((T[i], T[bestj]))
                used[i] = True
                used[bestj] = True

    # T-join edge set = symmetric difference of shortest paths
    join_edges = set()
    for a, b in pairs:
        path = _bfs_path(adj, a, b)
        if path is None:
            continue
        for k in range(len(path) - 1):
            e = (min(path[k], path[k + 1]), max(path[k], path[k + 1]))
            if e in join_edges:
                join_edges.discard(e)
            else:
                join_edges.add(e)

    # ensure join_edges subset of edges and G - join is Eulerian; else fallback
    if not join_edges.issubset(edges_set):
        join_edges = set()

    remaining = set(edges_set)
    adj2 = defaultdict(set)
    for u, v in remaining:
        adj2[u].add(v)
        adj2[v].add(u)

    for e in join_edges:
        u, v = e
        remaining.discard(e)
        adj2[u].discard(v)
        adj2[v].discard(u)

    cycles = _eulerian_cycles(adj2, n, remaining, deadline)

    pieces = cycles + [[u, v] for (u, v) in join_edges]
    # leftover (should be empty)
    for e in remaining:
        pieces.append(list(e))
    return pieces


def _verify(pieces, edges_set):
    used = set()
    for piece in pieces:
        if len(piece) == 2:
            e = (min(piece[0], piece[1]), max(piece[0], piece[1]))
            if e in used or e not in edges_set:
                return False
            used.add(e)
        elif len(piece) >= 3:
            if len(set(piece)) != len(piece):
                return False
            for i in range(len(piece)):
                u = piece[i]
                v = piece[(i + 1) % len(piece)]
                e = (min(u, v), max(u, v))
                if e in used or e not in edges_set:
                    return False
                used.add(e)
        else:
            return False
    return used == edges_set


def construct():
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        edges_set = set((min(u, v), max(u, v)) for u, v in edges)
        elapsed = time.time() - start_time
        remaining_time = _DEADLINE_SECONDS - elapsed
        if remaining_time < 3:
            results.append([[u, v] for u, v in edges])
            continue
        dl = start_time + min(remaining_time * 0.95, 45.0)
        pieces = _decompose(edges, n, dl)
        if not _verify(pieces, edges_set):
            pieces = _decompose_greedy_fallback(edges, n, dl)
            if not _verify(pieces, edges_set):
                pieces = [[u, v] for u, v in edges]
        results.append(pieces)
    return results


def _decompose_greedy_fallback(edges, n, deadline):
    remaining = set((min(u, v), max(u, v)) for u, v in edges)
    adj = defaultdict(set)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
    pieces = []
    while remaining:
        if time.time() > deadline:
            break
        best_cycle = None
        best_len = 0
        for start in range(n):
            if time.time() > deadline:
                break
            if len(adj.get(start, ())) < 2:
                continue
            c = _longest_cycle(adj, start, n, remaining, deadline)
            if c and len(c) > best_len:
                best_len = len(c)
                best_cycle = c
                if best_len >= n:
                    break
        if best_cycle:
            valid = True
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                if (min(u, v), max(u, v)) not in remaining:
                    valid = False
                    break
            if valid:
                pieces.append(best_cycle)
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    e = (min(u, v), max(u, v))
                    remaining.discard(e)
                    adj[u].discard(v)
                    adj[v].discard(u)
                continue
        e = next(iter(remaining))
        u, v = e
        remaining.discard(e)
        pieces.append([u, v])
        adj[u].discard(v)
        adj[v].discard(u)
    for e in remaining:
        pieces.append(list(e))
    return pieces
