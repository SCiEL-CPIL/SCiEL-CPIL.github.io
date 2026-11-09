import time
import random
from collections import defaultdict

_DEADLINE = 120.0  # wall-clock limit; keep safety margin
_START = time.time()

def _remaining_budget():
    return _DEADLINE - 3.0 - (time.time() - _START)

def instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_adjacency(edges):
    adj = defaultdict(set)
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_cycle(adj, edges_set, start, rng, max_nodes=200000):
    """DFS from start to find a cycle. Returns list of vertices (cycle) or None."""
    parent = {start: None}
    stack = [start]
    visited = {start}
    steps = 0
    while stack:
        if steps > max_nodes:
            return None
        steps += 1
        u = stack[-1]
        nbrs = list(adj[u])
        rng.shuffle(nbrs)
        found = False
        for v in nbrs:
            if (min(u, v), max(u, v)) not in edges_set:
                continue
            if v not in visited:
                visited.add(v)
                parent[v] = u
                stack.append(v)
                found = True
                break
            elif v != parent.get(u):
                # found a back edge u - v ; reconstruct cycle
                cycle = [u]
                cur = u
                while cur != v:
                    cur = parent[cur]
                    cycle.append(cur)
                # cycle now: u ... v, and closing edge v-u implied
                return cycle
        if not found:
            stack.pop()
    return None

def _decompose_one(n, p, seed):
    edges = instance(n, p, seed)
    edges_set = set(edges)
    adj = _build_adjacency(edges)
    rng = random.Random(12345 + seed)
    pieces = []

    while edges_set:
        if _remaining_budget() < 0.5:
            break
        # pick a start vertex with remaining degree >= 2
        starts = [u for u in adj if len(adj[u]) >= 2]
        if not starts:
            break
        starts.sort(key=lambda u: -len(adj[u]))
        # try a few starts to find a longer cycle
        best_cycle = None
        trial_starts = starts[:min(6, len(starts))]
        for s in trial_starts:
            cyc = _find_cycle(adj, edges_set, s, rng)
            if cyc is not None:
                if best_cycle is None or len(cyc) > len(best_cycle):
                    best_cycle = cyc
                    if len(best_cycle) >= 8:
                        break
        if best_cycle is None:
            # no cycle found via these starts; fall back to scanning all degree>=2
            for s in starts:
                if _remaining_budget() < 0.5:
                    break
                cyc = _find_cycle(adj, edges_set, s, rng)
                if cyc is not None:
                    best_cycle = cyc
                    break
        if best_cycle is None:
            # can't find cycles; dump remaining edges individually
            for e in list(edges_set):
                pieces.append(list(e))
            break

        # remove cycle edges
        cyc = best_cycle
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i + 1) % len(cyc)]
            e = (min(a, b), max(a, b))
            if e in edges_set:
                edges_set.discard(e)
                adj[a].discard(b)
                adj[b].discard(a)
        pieces.append(cyc)

    # leftover edges as individual pieces
    for e in list(edges_set):
        pieces.append(list(e))
    return pieces

def construct():
    _START_GLOBAL = time.time()
    global _START
    _START = _START_GLOBAL
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if _remaining_budget() < 1.0:
            # not enough time; produce trivial decomposition for remaining
            n, p, seed = spec
            results.append([list(e) for e in instance(n, p, seed)])
            continue
        results.append(_decompose_one(*spec))
    return results
