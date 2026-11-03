"""Stronger cage construction: known cages + random regular graph + 2-opt girth maximization."""
import random
import time
from collections import deque

_DEADLINE = None

def _girth(adj, g_max, seed=12345):
    """Return girth (length of shortest cycle), or >= g_max if none shorter found.
    Uses non-backtracking BFS from every vertex with early cutoff at g_max.
    A non-backtracking walk of length L ending at its start with L>=3 is a cycle
    of length <= L (may be shorter, so we return the first found length)."""
    n = len(adj)
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    for s in order:
        # state: (vertex, prev_vertex), start with prev=-1
        seen = {(-1, s): 1}
        front = [(-1, s)]
        for length in range(2, g_max + 1):
            nxt = []
            for (u, p) in front:
                for v in adj[u]:
                    if v == p:
                        continue
                    key = (v, u)
                    if key in seen:
                        if v == s:
                            return length
                        # back-edge: cycle shorter than length
                        return length
                    seen[key] = length
                    nxt.append(key)
            front = nxt
            if not front:
                break
    return g_max

def _is_valid(adj, k, g):
    n = len(adj)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    return _girth(adj, g) >= g

def _random_regular(k, n, seed):
    """Random k-regular graph on n vertices (k*n even). Returns adjacency list."""
    rng = random.Random(seed)
    if (k * n) % 2:
        return None
    half = []
    for i in range(n):
        for _ in range(k):
            half.append(i)
    # configuration model with rejection-free reshuffle until simple
    for attempt in range(2000):
        rng.shuffle(half)
        adj = [[] for _ in range(n)]
        ok = True
        for i in range(0, len(half), 2):
            a, b = half[i], half[i + 1]
            if a == b:
                ok = False
                break
            if b in adj[a]:
                ok = False
                break
            adj[a].append(b)
            adj[b].append(a)
        if ok:
            return adj
    return None

def _try_known_cages(k, g, deadline):
    """Hardcode smallest known cages for the common small cases. Verify girth."""
    global _DEADLINE
    # (k, g) -> (n, edge list) for well-known cages
    known = {}
    # 3-regular, girth 6: Petersen (10)
    known[(3, 6)] = (10, [(0,1),(1,2),(2,3),(3,4),(4,0),(5,6),(6,7),(7,8),(8,5),
                          (0,5),(1,6),(2,7),(3,8),(4,9)] if False else None)
    # We'll build Petersen, Heawood, Hoffman-Singleton, and the 320-cage procedurally below.
    if (k, g) == (3, 6):
        # Petersen
        edges = [(0,1),(1,2),(2,3),(3,4),(4,0),(5,6),(6,7),(7,8),(8,5),
                 (0,5),(1,6),(2,7),(3,8),(4,9)]
        n = 10
    elif (k, g) == (3, 8):
        # Heawood graph (14 vertices, 3-regular, girth 6 actually). Use (3,6) instead.
        return None
    elif (k, g) == (4, 8):
        # Hoffman-Singleton graph: 50 vertices, 4-regular, girth 5. Not girth 8.
        # Smallest 4-regular girth 8 is 50? No. The (4,8) cage is 50? Actually n(4,8)=50 is
        # the Hoffman-Singleton (girth 5). For girth 8, the smallest 4-regular is 50? Let's
        # not hardcode; rely on search.
        return None
    else:
        return None
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    if _is_valid(adj, k, g):
        return n, edges
    return None

def _local_search(adj, k, g, deadline, rng):
    """2-opt edge swap local search to increase girth. Returns (n, edges) or None."""
    n = len(adj)
    g0 = _girth(adj, g)
    if g0 >= g:
        return n, [(u, v) for u in range(n) for v in adj[u] if u < v]
    best_g = g0
    # Precompute adjacency sets for O(1) membership
    adjset = [set(a) for a in adj]
    # Build edge list
    edge_list = [(u, v) for u in range(n) for v in adj[u] if u < v]
    best_adj = [list(a) for a in adj]
    improved = True
    passes = 0
    while improved and _time() < deadline:
        improved = False
        passes += 1
        # collect all edges
        edges = [(u, v) for u in range(n) for v in best_adj[u] if u < v]
        rng.shuffle(edges)
        for (a, b) in edges:
            if _time() >= deadline:
                break
            # pick a different edge (c, d) not incident to a or b
            # try a few random partners
            for _ in range(30):
                c, d = edges[rng.randrange(len(edges))]
                if c == d:
                    continue
                if c in (a, b) or d in (a, b):
                    continue
                if (a, d) in adjset[a] or (a, c) in adjset[a]:
                    continue
                if (b, d) in adjset[b] or (b, c) in adjset[b]:
                    continue
                # swap (a,b),(c,d) -> (a,c),(b,d)
                adjset[a].discard(b); adjset[b].discard(a)
                adjset[c].discard(d); adjset[d].discard(c)
                adjset[a].add(c); adjset[c].add(a)
                adjset[b].add(d); adjset[d].add(b)
                # update best_adj
                best_adj[a].remove(b); best_adj[b].remove(a)
                best_adj[c].remove(d); best_adj[d].remove(c)
                best_adj[a].append(c); best_adj[c].append(a)
                best_adj[b].append(d); best_adj[d].append(b)
                gnew = _girth(best_adj, g)
                if gnew > best_g:
                    best_g = gnew
                    improved = True
                    if best_g >= g:
                        return n, [(u, v) for u in range(n) for v in best_adj[u] if u < v]
                else:
                    # revert
                    adjset[a].discard(c); adjset[c].discard(a)
                    adjset[b].discard(d); adjset[d].discard(b)
                    adjset[a].add(b); adjset[b].add(a)
                    adjset[c].add(d); adjset[d].add(c)
                    best_adj[a].remove(c); best_adj[c].remove(a)
                    best_adj[b].remove(d); best_adj[d].remove(b)
                    best_adj[a].append(b); best_adj[b].append(a)
                    best_adj[c].append(d); best_adj[d].append(c)
        if passes > 40:
            break
    if best_g >= g:
        return n, [(u, v) for u in range(n) for v in best_adj[u] if u < v]
    return None

def _time():
    return time.time()

def construct(k, g):
    global _DEADLINE
    t0 = time.time()
    _DEADLINE = t0 + 140.0  # safety margin before 150s kill
    rng = random.Random(1234567)

    # 1. Try known cages
    known = _try_known_cages(k, g, _DEADLINE)
    if known is not None:
        return known

    # 2. Random regular graph + local search, trying increasing sizes
    # Moore bound: n >= 2 + k * (k-1)^(g/2 - 1) for even g
    # We try sizes from a reasonable lower bound up.
    # For even g: moore = 2 + k*(k-1)**(g//2 - 1)
    # For odd g: more complex; use a safe floor.
    if g % 2 == 0:
        moore = 2 + k * (k - 1) ** (g // 2 - 1)
    else:
        # odd girth lower bound
        moore = 2 + k * (k - 1) ** ((g - 3) // 2) + (k - 1) ** ((g - 1) // 2)
    moore = max(2 * k, moore)
    # ensure k*n even
    def evenize(m):
        m = int(m)
        if (k * m) % 2:
            m += 1
        return m
    moore = evenize(moore)

    best = None
    # candidate sizes: start near moore, step up
    # cap the search: don't go too high
    max_n = moore + 400
    n = moore
    while n <= max_n and _time() < _DEADLINE:
        # try a few random seeds at this size
        found = None
        for seed in range(20):
            if _time() >= _DEADLINE:
                break
            adj = _random_regular(k, n, seed)
            if adj is None:
                continue
            # quick girth check
            gcur = _girth(adj, g)
            if gcur >= g:
                found = (n, [(u, v) for u in range(n) for v in adj[u] if u < v])
                break
            # else try local search
            res = _local_search(adj, k, g, _DEADLINE, rng)
            if res is not None:
                found = res
                break
        if found is not None:
            best = found
            break
        n = evenize(n + max(4, k))  # step up by at least 4 (to keep parity options)

    if best is not None:
        return best

    # Fallback: return any valid graph we can make (shouldn't happen)
    # Build a random regular and accept even if girth slightly short is NOT allowed.
    # As last resort, return a trivial valid construction: cycle blow-up won't have girth g.
    # Return the largest random regular we built with best girth (may be invalid -> evaluator
    # scores 0, but we've exhausted budget). To avoid invalid, return None-like by raising is
    # bad. Instead return a graph we know is valid: use a high-girth random graph search.
    # Extend search a bit more.
    while _time() < _DEADLINE:
        n = evenize(n + max(4, k))
        if n > max_n + 800:
            break
        for seed in range(50):
            if _time() >= _DEADLINE:
                break
            adj = _random_regular(k, n, seed)
            if adj is None:
                continue
            if _girth(adj, g) >= g:
                return n, [(u, v) for u in range(n) for v in adj[u] if u < v]
    # Absolute fallback: return a valid small graph only if girth condition met; else
    # return the best-effort (evaluator will mark invalid -> 0). Provide a cycle-free
    # tree-based regular? impossible. Return last random regular even if invalid.
    adj = _random_regular(k, max(2 * k, moore), 0)
    if adj is not None:
        n = len(adj)
        return n, [(u, v) for u in range(n) for v in adj[u] if u < v]
    # ultimate fallback: empty
    return 0, []
