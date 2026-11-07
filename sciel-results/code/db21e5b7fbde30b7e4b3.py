import time
import random

_N = 15
_ALL = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
_START = time.time()
_DEADLINE = _START + 50.0


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


def _norm(e):
    a, b = e
    return (a, b) if a < b else (b, a)


def _count_violations(g):
    """Total number of violating simple 3-edge paths (exact)."""
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


def _solve_once(rng):
    # Start with K_7,8
    # Vertices 0-6 in A, 7-14 in B
    g = set()
    for u in range(7):
        for v in range(7, 15):
            g.add((u, v))
    
    # Phase 1: Greedy add edges from remaining
    remaining = [e for e in _ALL if e not in g]
    rng.shuffle(remaining)
    for e in remaining:
        if time.time() > _DEADLINE:
            break
        g2 = g | {e}
        if _validity(g2)[0]:
            g = g2

    # Phase 2: Local search polish
    adj = [set() for _ in range(_N)]
    deg = [0] * _N
    for (u, v) in g:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1

    def edge_violations(u, v):
        a = [deg[x] for x in adj[u] if x != v]
        b = [deg[x] for x in adj[v] if x != u]
        ca = {}
        for d in a:
            ca[d] = ca.get(d, 0) + 1
        cb = {}
        for d in b:
            cb[d] = cb.get(d, 0) + 1
        tot = 0
        for d, cnt in ca.items():
            if d in cb:
                tot += cnt * cb[d]
        return tot

    def add_edge(u, v):
        g.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1

    def remove_edge(u, v):
        g.discard((u, v))
        adj[u].discard(v)
        adj[v].discard(u)
        deg[u] -= 1
        deg[v] -= 1

    best_g = set(g)
    best_sz = len(g)
    tabu = {}
    tabu_counter = 0
    TABU_TENURE = 12
    no_improve = 0
    MAX_POLISH = 4000

    for _ in range(MAX_POLISH):
        if time.time() > _DEADLINE:
            break
        moves = []
        for e in _ALL:
            if e in g:
                moves.append((e, None))
            else:
                for f in g:
                    moves.append((e, f))
        rng.shuffle(moves)
        best_delta = None
        best_move = None
        for e, f in moves:
            if f is None:
                if tabu.get(e, -1) >= tabu_counter and len(g) > best_sz:
                    continue
                delta = -edge_violations(*e)
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_move = (e, f)
            else:
                if tabu.get(e, -1) >= tabu_counter or tabu.get(f, -1) >= tabu_counter:
                    if len(g) > best_sz:
                        continue
                delta = -edge_violations(*e) + edge_violations(*f)
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_move = (e, f)

        if best_move is None:
            break
        e, f = best_move
        if f is not None:
            remove_edge(*f)
            tabu[f] = tabu_counter + TABU_TENURE
        add_edge(*e)
        tabu[e] = tabu_counter + TABU_TENURE
        tabu_counter += 1

        if len(g) > best_sz:
            best_sz = len(g)
            best_g = set(g)
            no_improve = 0
        else:
            no_improve += 1
        if no_improve > 60:
            break

    if len(best_g) >= len(g):
        g = best_g
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
            if len(g) > best_sz:
                best_sz = len(g)
                best = list(g)
        restart += 1
    if best is None:
        # Guaranteed-valid fallback: K_7,8
        best = [(u, v) for u in range(7) for v in range(7, 15)]
    out = [[int(u), int(v)] for (u, v) in best]
    return out
