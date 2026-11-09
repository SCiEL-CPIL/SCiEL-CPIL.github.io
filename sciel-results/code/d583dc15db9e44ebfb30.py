"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_best_cycle(adj, n, rng):
    """Iterative randomized DFS. Returns the longest cycle found (any back-edge
    to an ancestor on the current path yields a cycle)."""
    candidates = [v for v in range(n) if len(adj[v]) >= 2]
    if not candidates:
        return None

    candidates.sort(key=lambda x: len(adj[x]), reverse=True)
    starts = list(dict.fromkeys(candidates[:5] + (rng.sample(candidates, min(5, len(candidates))) if len(candidates) > 5 else [])))
    rng.shuffle(starts)

    best = None
    best_len = 0
    for start in starts:
        color = {start: 1}
        parent = {start: -1}
        path = [start]
        pidx = {start: 0}
        local = None
        local_len = 0
        stack = [(start, iter(rng.sample(list(adj[start]), len(adj[start]))))]
        while stack:
            u, it = stack[-1]
            advanced = False
            for w in it:
                if w == parent[u]:
                    continue
                if color.get(w, 0) == 0:
                    color[w] = 1
                    parent[w] = u
                    pidx[w] = len(path)
                    path.append(w)
                    stack.append((w, iter(rng.sample(list(adj[w]), len(adj[w])))))
                    advanced = True
                    break
                elif color.get(w, 0) == 1:
                    idx = pidx[w]
                    cyc = path[idx:]
                    if len(cyc) > local_len:
                        local_len = len(cyc)
                        local = list(cyc)
            if not advanced:
                color[u] = 2
                stack.pop()
                if path:
                    path.pop()
        if local and local_len > best_len:
            best = local
            best_len = local_len
        if best_len >= n:
            break
    return best

def _decompose_once(edges, n, rng):
    """One randomized greedy decomposition: repeatedly extract the longest cycle."""
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    pieces = []
    while True:
        if not any(adj[v] for v in range(n)):
            break
        cyc = _find_best_cycle(adj, n, rng)
        if cyc is None or len(cyc) < 3:
            break
        for j in range(len(cyc)):
            a, b = cyc[j], cyc[(j + 1) % len(cyc)]
            if b in adj[a]:
                adj[a].discard(b)
            if a in adj[b]:
                adj[b].discard(a)
        pieces.append(list(cyc))
    for v in range(n):
        for w in sorted(adj[v]):
            if v < w:
                pieces.append([v, w])
    return pieces

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for spec in specs:
        edges = _instance(*spec)
        n = spec[0]
        best = None
        best_count = None
        # restart 0 gives a guaranteed valid baseline
        best = _decompose_once(edges, n, Random(42 + spec[2]))
        best_count = len(best)
        # multi-restart best-of search
        r = 1
        while time.time() < deadline:
            cand = _decompose_once(edges, n, Random(1000 + r))
            if len(cand) < best_count:
                best = cand
                best_count = len(cand)
            r += 1
        results.append(best)
    return results
# EVOLVE-BLOCK-END
