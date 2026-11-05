import random
import time
from itertools import combinations

N = 20
C = 20
DEADLINE = 118.0  # safety margin well before the 150 s wall-clock kill

_START = time.time()


def _time_left():
    return DEADLINE - (time.time() - _START)


# Precompute all 4-cycles (3 cyclic orders per 4-set) once.
def _precompute_cycles(n):
    cycles = []
    for combo in combinations(range(n), 4):
        a, b, c, d = combo
        cycles.append(((a, b), (b, c), (c, d), (d, a)))
        cycles.append(((a, b), (b, d), (d, c), (c, a)))
        cycles.append(((a, c), (c, b), (b, d), (d, a)))
    return cycles


_CYCLES = _precompute_cycles(N)


def _validate(edges):
    """Return True iff every 4-cycle fully present is rainbow."""
    edge_set = {}
    for u, v, c in edges:
        edge_set[(u, v)] = c
        edge_set[(v, u)] = c
    for cyc in _CYCLES:
        cols = []
        ok = True
        for (u, v) in cyc:
            c = edge_set.get((u, v))
            if c is None:
                ok = False
                break
            cols.append(c)
        if ok and len(set(cols)) != 4:
            return False
    return True


def _greedy_build(seed, initial_edges=None, order_type='random'):
    """Build a valid coloured graph, optionally seeded from initial_edges."""
    rng = random.Random(seed)
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]

    adj = [[-1] * N for _ in range(N)]
    neighbors = [set() for _ in range(N)]
    edges = []

    if initial_edges:
        existing = set()
        for u, v, c in initial_edges:
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            edges.append([u, v, c])
            existing.add((min(u, v), max(u, v)))
        remaining = [e for e in all_edges if (min(e[0], e[1]), max(e[0], e[1])) not in existing]
    else:
        remaining = all_edges[:]

    if order_type == 'random':
        rng.shuffle(remaining)
    elif order_type == 'reverse':
        remaining = remaining[::-1]
    # else natural order

    for (u, v) in remaining:
        impossible = False
        forbidden = set()
        for a in neighbors[u]:
            if a == v:
                continue
            ca = adj[u][a]
            for b in neighbors[v]:
                if b == u or b == a:
                    continue
                if adj[a][b] != -1:
                    cb = adj[a][b]
                    cv = adj[b][v]
                    if ca == cb or cb == cv or ca == cv:
                        impossible = True
                        break
                    forbidden.add(ca)
                    forbidden.add(cb)
                    forbidden.add(cv)
            if impossible:
                break
        if not impossible:
            # prefer least-used colour among neighbours of u and v
            color_freq = {}
            for a in neighbors[u]:
                if a != v:
                    ca = adj[u][a]
                    color_freq[ca] = color_freq.get(ca, 0) + 1
            for b in neighbors[v]:
                if b != u:
                    cb = adj[v][b]
                    color_freq[cb] = color_freq.get(cb, 0) + 1
            available = [c for c in range(C) if c not in forbidden]
            if available:
                rng.shuffle(available)
                available.sort(key=lambda c: color_freq.get(c, 0))
                c = available[0]
                adj[u][v] = c
                adj[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
    return edges


def _bipartite_seed(rng):
    """K_{10,10} core with a linear colouring, removing edges to fix bad 4-cycles."""
    A = list(range(10))
    B = list(range(10, 20))
    a = rng.randint(0, 19)
    b = rng.randint(1, 19)
    c0 = rng.randint(0, 19)

    def col(u, v):
        return (a * u + b * v + c0) % 20

    ec = {}
    for u in A:
        for v in B:
            ec[(u, v)] = col(u, v)
            ec[(v, u)] = col(u, v)

    bad = []
    for i in range(10):
        for j in range(i + 1, 10):
            u1, u2 = A[i], A[j]
            for k in range(10):
                for l in range(k + 1, 10):
                    v1, v2 = B[k], B[l]
                    cols = {ec[(u1, v1)], ec[(u2, v1)], ec[(u2, v2)], ec[(u1, v2)]}
                    if len(cols) < 4:
                        bad.append([(u1, v1), (u2, v1), (u2, v2), (u1, v2)])

    remove = set()
    rem = bad[:]
    while rem and _time_left() > 15.0:
        cands = set()
        for cyc in rem:
            for e in cyc:
                if e not in remove:
                    cands.add(e)
        if not cands:
            break
        best_e, best_score = None, -1
        for e in cands:
            hits = sum(1 for cyc in rem if e in cyc)
            score = hits + rng.random() * 0.5
            if score > best_score:
                best_score = score
                best_e = e
        if best_e is None:
            break
        remove.add(best_e)
        rem = [cyc for cyc in rem if best_e not in cyc]

    out = []
    for u in A:
        for v in B:
            if (u, v) not in remove:
                out.append([u, v, ec[(u, v)]])
    return out


def _try_add_missing(current):
    """Try to add one missing edge with a valid colour; return new list or None."""
    adj = [[-1] * N for _ in range(N)]
    neighbors = [set() for _ in range(N)]
    existing = set()
    for u, v, c in current:
        adj[u][v] = c
        adj[v][u] = c
        neighbors[u].add(v)
        neighbors[v].add(u)
        existing.add((min(u, v), max(u, v)))

    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    missing = [e for e in all_edges if (min(e[0], e[1]), max(e[0], e[1])) not in existing]
    if not missing:
        return None

    rng = random.Random()
    rng.shuffle(missing)
    for (u, v) in missing:
        impossible = False
        forbidden = set()
        for a in neighbors[u]:
            if a == v:
                continue
            ca = adj[u][a]
            for b in neighbors[v]:
                if b == u or b == a:
                    continue
                if adj[a][b] != -1:
                    cb = adj[a][b]
                    cv = adj[b][v]
                    if ca == cb or cb == cv or ca == cv:
                        impossible = True
                        break
                    forbidden.add(ca)
                    forbidden.add(cb)
                    forbidden.add(cv)
            if impossible:
                break
        if not impossible:
            available = [c for c in range(C) if c not in forbidden]
            if available:
                c = rng.choice(available)
                return current + [[u, v, c]]
    return None


def construct():
    best_edges = []
    best_count = 0

    # Phase 1: random greedy restarts over several orderings
    for ot in ('random', 'natural', 'reverse'):
        for i in range(60):
            if _time_left() < 20.0:
                break
            edges = _greedy_build(i * 7 + 1, None, ot)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

    # Phase 2: bipartite-seeded greedy
    master = random.Random(20240517)
    for i in range(20):
        if _time_left() < 20.0:
            break
        seed_edges = _bipartite_seed(master)
        edges = _greedy_build(1000 + i, seed_edges, 'random')
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges

    # Phase 3: local search (add missing edges, perturb + rebuild)
    current = best_edges[:]
    it = 0
    while _time_left() > 10.0:
        it += 1
        move_type = random.random()
        if move_type < 0.6:
            new_edges = _try_add_missing(current)
            if new_edges is not None:
                if len(new_edges) > best_count:
                    best_count = len(new_edges)
                    best_edges = new_edges
                    current = new_edges[:]
        else:
            k = random.randint(1, 3)
            if len(current) > k:
                to_remove = random.sample(current, k)
                rset = set((min(u, v), max(u, v)) for u, v, _ in to_remove)
                base = [e for e in current if (min(e[0], e[1]), max(e[0], e[1])) not in rset]
                new_edges = _greedy_build(it * 131 + random.randint(0, 100), base, 'random')
                if len(new_edges) > best_count:
                    best_count = len(new_edges)
                    best_edges = new_edges
                    current = new_edges[:]

    if best_edges and _validate(best_edges):
        return best_edges
    # last-resort valid graph (a star has no 4-cycles)
    return [[0, v, 0] for v in range(1, N)]
