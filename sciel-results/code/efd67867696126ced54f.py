import time
import heapq
from collections import defaultdict
from random import Random

_START = time.time()
_LIMIT = 150.0
_MARGIN = 2.0

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose(edges):
    n = max((max(e) for e in edges), default=0) + 1
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # connected component containing vertex 0
    seen = {0}
    stack = [0]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    if len(seen) != n:
        # fall back to trivial per-edge paths
        return [[list(e)] for e in edges]

    V = n
    SRC = 2 * V
    SNK = 2 * V + 1
    N = SNK + 1
    INF = 10 ** 9

    def in_(v):
        return 2 * v

    def out_(v):
        return 2 * v + 1

    # build residual graph: adj[node] -> list of [to, cap, cost, rev]
    adj2 = defaultdict(list)
    def add_edge(a, b, cap, cost):
        adj2[a].append([b, cap, cost, len(adj2[b])])
        adj2[b].append([a, 0, -cost, len(adj2[a]) - 1])

    # vertex capacity arcs
    for v in range(V):
        add_edge(in_(v), out_(v), 1, 0)
    # edge arcs
    for u, v in edges:
        add_edge(out_(u), in_(v), 1, 1)
        add_edge(out_(v), in_(u), 1, 1)
    # super source / sink
    for v in range(V):
        add_edge(SRC, in_(v), 1, 0)
        add_edge(out_(v), SNK, 1, 0)

    # potentials via Bellman-Ford from SRC
    dist = [INF] * N
    dist[SRC] = 0
    for _ in range(N):
        changed = False
        for a in adj2:
            da = dist[a]
            if da == INF:
                continue
            for b, cap, cost, rev in adj2[a]:
                if cap > 0 and da + cost < dist[b]:
                    dist[b] = da + cost
                    changed = True
        if not changed:
            break
    for i in range(N):
        if dist[i] == INF:
            dist[i] = 0
    pot = dist

    paths = []
    while time.time() < _LIMIT - _MARGIN:
        # Dijkstra with potentials
        d = [INF] * N
        d[SRC] = 0
        pv = [-1] * N
        pe = [-1] * N
        heap = [(0, SRC)]
        while heap:
            du, u = heapq.heappop(heap)
            if du > d[u]:
                continue
            for idx, (v, cap, cost, rev) in enumerate(adj2[u]):
                if cap <= 0:
                    continue
                nd = du + cost + pot[u] - pot[v]
                if nd < d[v]:
                    d[v] = nd
                    pv[v] = u
                    pe[v] = idx
                    heapq.heappush(heap, (nd, v))
        if d[SNK] == INF:
            break
        for i in range(N):
            if d[i] < INF:
                pot[i] += d[i]
        # augment 1 unit along path
        v = SNK
        while v != SRC:
            u = pv[v]
            e = pe[v]
            adj2[u][e][1] -= 1
            adj2[v][e[3]][1] += 1
            v = u
        # extract path vertices from SRC->...->SNK
        v = SNK
        chain = []
        while v != SRC:
            chain.append(v)
            v = pv[v]
        chain.reverse()
        verts = []
        for node in chain:
            if SRC < node < SNK:
                if node % 2 == 0:
                    verts.append(node // 2)
                else:
                    verts.append((node - 1) // 2)
        verts = list(dict.fromkeys(verts))
        if len(verts) >= 2:
            paths.append(verts)
    return paths

def construct():
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    for spec in specs:
        if time.time() > _LIMIT - _MARGIN:
            break
        edges = _instance(*spec)
        paths = _decompose(edges)
        results.append(paths)
    # fill any missing graphs with trivial per-edge paths
    while len(results) < len(specs):
        spec = specs[len(results)]
        results.append([[list(e)] for e in _instance(*spec)])
    return results
