"""Cage n(4,10): construct a 4-regular graph with girth >= 10 on as few vertices as possible.

Primary: voltage lift of a 4-regular 40-vertex base over Z_8 -> 320 vertices, searched
and verified for girth >= 10 via non-backtracking BFS. Safety net: if no verified 320-vertex
graph is found in time, fall back to the Lazebnik-Ustimenko-Woldar CD(5,4) graph (valid,
4-regular, girth >= 10), guaranteeing a correct non-zero score.
"""

# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque


def _make_base_4regular(nv, seed):
    rng = random.Random(seed)
    for _ in range(300):
        half = [v for v in range(nv) for _ in range(4)]
        rng.shuffle(half)
        edges, seen, ok = [], set(), True
        for i in range(0, len(half), 2):
            a, b = half[i], half[i + 1]
            if a == b:
                ok = False
                break
            u, v = (a, b) if a < b else (b, a)
            if (u, v) in seen:
                ok = False
                break
            seen.add((u, v))
            edges.append((u, v))
        if ok:
            deg = defaultdict(int)
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if all(deg[v] == 4 for v in range(nv)):
                return edges
    return None


def _lift(base_edges, voltages, group_size):
    n = 40 * group_size
    adj = [[] for _ in range(n)]
    for (u, v), vol in zip(base_edges, voltages):
        vol %= group_size
        for c in range(group_size):
            a = u * group_size + c
            b = v * group_size + ((c + vol) % group_size)
            adj[a].append(b)
            adj[b].append(a)
    return n, adj


def _girth_ok(n, adj, g, deadline):
    for i in range(n):
        if len(adj[i]) != 4 or len(set(adj[i])) != 4:
            return False
    for s in range(n):
        dist = {}
        q = deque()
        q.append((s, -1, 0))
        dist[(s, -1)] = 0
        while q:
            v, prev, d = q.popleft()
            if d >= g - 1:
                continue
            nd = d + 1
            for w in adj[v]:
                if w == prev:
                    continue
                if w == s and nd >= 3:
                    return False
                state = (w, v)
                if state in dist:
                    continue
                dist[state] = nd
                q.append((w, v, nd))
            if time.time() > deadline:
                return False
    return True


def _lzuw_fallback(k, g):
    """Lazebnik-Ustimenko-Woldar CD(d,q) with q=k, girth >= d+5 (odd d). Valid 4-regular."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add, sub, mul = (lambda x,y:x^y),(lambda x,y:x^y),(lambda x,y:table[x][y])
    else:
        add, sub, mul = (lambda x,y:(x+y)%q),(lambda x,y:(x-y)%q),(lambda x,y:x*y%q)
    rule = {2:(1,1),3:(2,1),4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)

    start = (0, (None,)+(0,)*d)
    index, order, edges = {start:0}, [start], []
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
    deadline = start_time + 110.0

    group_size = 8
    nv = 40
    best = None
    base_cache = {}
    base_id = 0
    while time.time() < deadline:
        if base_id in base_cache:
            base_edges = base_cache[base_id]
        else:
            base_edges = _make_base_4regular(nv, base_id)
            base_cache[base_id] = base_edges
        if base_edges is not None:
            m = len(base_edges)
            for volt_seed in range(10):
                if time.time() > deadline:
                    break
                rng = random.Random(700000 + base_id*1000 + volt_seed)
                voltages = [rng.randrange(group_size) for _ in range(m)]
                n, adj = _lift(base_edges, voltages, group_size)
                if _girth_ok(n, adj, g, deadline):
                    best = (n, adj)
                    break
        if best is not None:
            break
        base_id += 1

    if best is not None:
        n, adj = best
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return n, edges

    # Safety net: guaranteed valid 4-regular girth>=10 graph.
    return _lzuw_fallback(k, g)
# EVOLVE-BLOCK-END
