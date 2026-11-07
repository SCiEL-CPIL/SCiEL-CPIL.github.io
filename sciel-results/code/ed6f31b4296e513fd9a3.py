import random
import time
from collections import defaultdict

_N = 15
_ALL = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
_START = time.time()
_DEADLINE = _START + 80.0


def _validity(g):
    """Return (is_valid, witness_quadruple_or_None)."""
    deg = [0] * _N
    adj = [set() for _ in range(_N)]
    for (u, v) in g:
        deg[u] += 1
        deg[v] += 1
        adj[u].add(v)
        adj[v].add(u)
    for (u, v) in g:
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if deg[s] == deg[t]:
                    return (False, (s, u, v, t))
    return (True, None)


def _count_violations(g):
    """Count total number of violating 3-edge paths."""
    deg = [0] * _N
    adj = [set() for _ in range(_N)]
    for (u, v) in g:
        deg[u] += 1
        deg[v] += 1
        adj[u].add(v)
        adj[v].add(u)
    total = 0
    for (u, v) in g:
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if deg[s] == deg[t]:
                    total += 1
    return total


def _norm(e):
    a, b = e
    return (a, b) if a < b else (b, a)


def _solve_once(rng):
    """Start from complete graph, greedily remove edges to reach valid state, then re-add."""
    g = set(_ALL)
    # Greedy edge removal to reach a valid graph.
    while True:
        if time.time() > _DEADLINE:
            # Fall back to a known-valid sparse graph (a star) to avoid timeout.
            return [(0, v) for v in range(1, _N)]
        ok, w = _validity(g)
        if ok:
            break
        s, u, v, t = w
        cand = [_norm((s, u)), _norm((u, v)), _norm((v, t))]
        rng.shuffle(cand)
        best = None
        best_cnt = None
        for e in cand:
            g2 = g - {e}
            c = _count_violations(g2)
            if best_cnt is None or c < best_cnt:
                best_cnt = c
                best = e
        if best_cnt is None or best_cnt >= len(g):
            best = cand[0]
        g.discard(best)
    
    # Re-add every edge that preserves validity.
    for e in list(_ALL):
        if time.time() > _DEADLINE:
            break
        if e not in g:
            g2 = g | {e}
            if _validity(g2)[0]:
                g = g2
    return g


def _local_search(g, rng):
    """Try to improve by removing one edge and re-adding multiple edges."""
    g = set(g)
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        # Try removing each edge and see if we can re-add more edges
        edges = list(g)
        rng.shuffle(edges)
        for e in edges:
            if time.time() > _DEADLINE:
                return g
            g2 = g - {e}
            # Try to re-add all missing edges
            for f in _ALL:
                if f not in g2:
                    g3 = g2 | {f}
                    if _validity(g3)[0]:
                        g2 = g3
            if len(g2) > len(g):
                g = g2
                improved = True
                break
    return g


def construct():
    best = None
    best_sz = -1
    base_seed = 12345
    restart = 0
    while time.time() < _DEADLINE:
        rng = random.Random(base_seed + 1000 * restart)
        g = _solve_once(rng)
        if _validity(set(g))[0]:
            # Try local search to improve
            g2 = _local_search(g, rng)
            if _validity(set(g2))[0] and len(g2) > len(g):
                g = g2
            if len(g) > best_sz:
                best_sz = len(g)
                best = list(g)
        restart += 1
    
    if best is None:
        # Guaranteed-valid fallback: a star is always valid
        best = [(0, v) for v in range(1, _N)]
    
    # Coerce to exact Python ints and plain list-of-lists.
    out = [[int(u), int(v)] for (u, v) in best]
    return out


# EVOLVE-BLOCK-END
