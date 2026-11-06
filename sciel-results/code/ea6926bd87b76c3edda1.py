import time
import random
import sys

N = 20
C = 20
DEADLINE = 138.0  # safety margin under the 150s wall-clock limit


def _build_hadamard(n):
    """Sylvester Hadamard of size n (n must be a power of 2)."""
    H = [[1]]
    while len(H) < n:
        H = [[a + b for a, b in zip(r, r)] for r in H] + \
            [[a - b for a, b in zip(r, r)] for r in H]
    return H


def _make_adj(edges):
    adj = [[-1] * N for _ in range(N)]
    neigh = [set() for _ in range(N)]
    for u, v, c in edges:
        adj[u][v] = c
        adj[v][u] = c
        neigh[u].add(v)
        neigh[v].add(u)
    return adj, neigh


def _validate(edges):
    """Return True iff every 4-cycle is rainbow (4 distinct colours)."""
    adj, neigh = _make_adj(edges)
    for u in range(N):
        for v in neigh[u]:
            if u >= v:
                continue
            common = [w for w in neigh[u] if adj[v][w] >= 0 and w != u]
            for i in range(len(common)):
                x = common[i]
                for j in range(i + 1, len(common)):
                    y = common[j]
                    cols = {adj[u][x], adj[x][v], adj[v][y], adj[y][u]}
                    if len(cols) != 4:
                        return False
    return True


def _count_violations(adj, neigh):
    violations = 0
    for u in range(N):
        for v in neigh[u]:
            if u >= v:
                continue
            common = [w for w in neigh[u] if adj[v][w] >= 0 and w != u]
            for i in range(len(common)):
                x = common[i]
                for j in range(i + 1, len(common)):
                    y = common[j]
                    cols = {adj[u][x], adj[x][v], adj[v][y], adj[y][u]}
                    if len(cols) != 4:
                        violations += 1
    return violations


def _analyze_edge(u, v, adj, neigh):
    """For candidate edge (u,v): return (impossible, forbidden_set).
    impossible  -> some existing u-x-y-v path already has a colour clash
                   (the new edge can never fix it).
    forbidden   -> colours the new edge must avoid to stay rainbow.
    """
    forbidden = set()
    impossible = False
    for x in neigh[u]:
        if x == v:
            continue
        cux = adj[u][x]
        for y in neigh[v]:
            if y == u or y == x:
                continue
            if adj[x][y] < 0:
                continue
            cxy = adj[x][y]
            cyv = adj[y][v]
            if cux == cxy or cxy == cyv or cux == cyv:
                return True, forbidden
            forbidden.add(cux)
            forbidden.add(cxy)
            forbidden.add(cyv)
    return False, forbidden


def _add_edge(edges, adj, neigh, u, v, c):
    edges.append([u, v, c])
    adj[u][v] = c
    adj[v][u] = c
    neigh[u].add(v)
    neigh[v].add(u)


def _remove_edge(edges, adj, neigh, u, v):
    for i, (a, b, cc) in enumerate(edges):
        if {a, b} == {u, v}:
            edges.pop(i)
            break
    adj[u][v] = -1
    adj[v][u] = -1
    neigh[u].discard(v)
    neigh[v].discard(u)


def _try_add(edges, adj, neigh, u, v, rng):
    impossible, forbidden = _analyze_edge(u, v, adj, neigh)
    if impossible:
        return False
    valid = [c for c in range(C) if c not in forbidden]
    if not valid:
        return False
    rng.shuffle(valid)
    _add_edge(edges, adj, neigh, u, v, valid[0])
    return True


def _greedy_hadamard(seed, deadline):
    """Spectral (Hadamard) base colouring + degree-sorted greedy build."""
    rng = random.Random(seed)
    H = _build_hadamard(N)
    perm = list(range(N))
    rng.shuffle(perm)
    color_of_vertex = [0] * N
    for i in range(N):
        color_of_vertex[perm[i]] = (H[perm[i]][perm[i]] + 1) % C

    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    edges = []
    adj, neigh = _make_adj(edges)

    # Sort by current degree (ascending): low-conflict edges first.
    order = sorted(all_pairs, key=lambda p: (len(neigh[p[0]]) + len(neigh[p[1]]), p))

    added = True
    while added and time.time() < deadline:
        added = False
        for u, v in order:
            if adj[u][v] >= 0:
                continue
            if time.time() > deadline:
                break
            if _try_add(edges, adj, neigh, u, v, rng):
                added = True
                break  # degrees changed -> resort next pass
        if added:
            order = sorted(all_pairs, key=lambda p: (len(neigh[p[0]]) + len(neigh[p[1]]), p))
    return edges


def _greedy_random(seed, deadline):
    rng = random.Random(seed)
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    rng.shuffle(all_pairs)
    edges = []
    adj, neigh = _make_adj(edges)
    for u, v in all_pairs:
        if time.time() > deadline:
            break
        _try_add(edges, adj, neigh, u, v, rng)
    return edges


def _local_search(edges, deadline):
    """Refine: add missing edges, then recolor/remove to unlock more."""
    rng = random.Random(12345)
    edges = [e[:] for e in edges]
    adj, neigh = _make_adj(edges)

    def missing():
        return [(u, v) for u in range(N) for v in range(u + 1, N) if adj[u][v] < 0]

    # Phase A: greedy fill of missing edges.
    while time.time() < deadline:
        miss = missing()
        if not miss:
            break
        progress = False
        for u, v in miss:
            if _try_add(edges, adj, neigh, u, v, rng):
                progress = True
        if not progress:
            break

    # Phase B: recolor / remove to break impossible paths, then re-fill.
    for _ in range(4000):
        if time.time() > deadline:
            break
        miss = missing()
        if not miss:
            break
        u, v = rng.choice(miss)
        impossible, _ = _analyze_edge(u, v, adj, neigh)
        if not impossible:
            break
        # Find a conflicting u-x-y-v path.
        target = None
        for x in neigh[u]:
            if x == v:
                continue
            for y in neigh[v]:
                if y == u or y == x:
                    continue
                if adj[x][y] < 0:
                    continue
                cux, cxy, cyv = adj[u][x], adj[x][y], adj[y][v]
                if cux == cxy or cxy == cyv or cux == cyv:
                    target = (x, y)
                    break
            if target:
                break
        if not target:
            break
        x, y = target
        candidates = [(u, x), (x, y), (y, v)]
        rng.shuffle(candidates)
        improved = False
        for a, b in candidates:
            old = adj[a][b]
            for nc in range(C):
                if nc == old:
                    continue
                # tentatively recolor
                adj[a][b] = nc
                adj[b][a] = nc
                # quick check: does (a,b) stay rainbow with all its cycles?
                ok = True
                for w in neigh[a]:
                    if w == b:
                        continue
                    for z in neigh[b]:
                        if z == a or z == w:
                            continue
                        if adj[w][z] < 0:
                            continue
                        cols = {nc, adj[a][w], adj[w][z], adj[z][b]}
                        if len(cols) != 4:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    for i, (pa, pb, cc) in enumerate(edges):
                        if {pa, pb} == {a, b}:
                            edges[i][2] = nc
                            break
                    improved = True
                    break
                adj[a][b] = old
                adj[b][a] = old
            if improved:
                break
        if not improved:
            # remove the middle edge of the path to unblock
            for i, (pa, pb, cc) in enumerate(edges):
                if {pa, pb} == {x, y}:
                    edges.pop(i)
                    break
            adj[x][y] = -1
            adj[y][x] = -1
            neigh[x].discard(y)
            neigh[y].discard(x)
            # re-fill a bit
            for uu, vv in missing():
                _try_add(edges, adj, neigh, uu, vv, rng)
    return edges


def construct():
    start = time.time()
    budget = start + DEADLINE
    best = []
    best_len = 0

    def consider(edges):
        nonlocal best, best_len
        if len(edges) > best_len:
            best = [e[:] for e in edges]
            best_len = len(edges)

    # Phase 1: diverse seeds (Hadamard spectral + random greedy).
    seed_budget = budget - 30.0
    for s in range(60):
        if time.time() > seed_budget:
            break
        if s % 2 == 0:
            e = _greedy_hadamard(s, min(budget - 2.0, time.time() + 1.2))
        else:
            e = _greedy_random(s, min(budget - 2.0, time.time() + 1.2))
        consider(e)

    # Phase 2: local-search refinement on the current best (and a few others).
    ls_budget = budget - 3.0
    if best:
        improved = _local_search(best, ls_budget)
        consider(improved)

    # Phase 3: a couple of fresh restarts + refinement if time remains.
    for s in range(60, 90):
        if time.time() > budget - 8.0:
            break
        e = _greedy_hadamard(1000 + s, min(budget - 4.0, time.time() + 1.0))
        consider(e)
        if time.time() < ls_budget:
            consider(_local_search(best, ls_budget))

    if not best:
        best = [[0, v, 0] for v in range(1, N)]  # safe star fallback
    return best


if __name__ == "__main__":
    res = construct()
    print("edges:", len(res))
    print("valid:", _validate(res))
