"""Triangle-free diameter-four augmentation: branch-and-bound exact search + constructive heuristic."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10],
                        [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18],
                        [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def _bfs(n, adj):
    """Return list of BFS distance lists from each source; -1 for unreachable."""
    dist = []
    for s in range(n):
        d = [-1] * n
        d[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            nd = d[u] + 1
            for v in adj[u]:
                if d[v] < 0:
                    d[v] = nd
                    q.append(v)
        dist.append(d)
    return dist


def _diam_from_dist(dist, n):
    best = 0
    for i in range(n):
        row = dist[i]
        for j in range(i + 1, n):
            if row[j] < 0:
                return -1
            if row[j] > best:
                best = row[j]
    return best


def _base_adj(n, base_edges):
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _constructive(n, base_edges):
    """Greedy diameter reduction; returns a valid solution (list of (u,v) tuples)."""
    base_set = set()
    for u, v in base_edges:
        base_set.add((u, v)); base_set.add((v, u))
    even = [i for i in range(n) if i % 2 == 0]
    odd = [i for i in range(n) if i % 2 == 1]
    cands = [(u, v) for u in even for v in odd if (u, v) not in base_set]

    def diam(added):
        adj = _base_adj(n, base_edges)
        for u, v in added:
            adj[u].add(v); adj[v].add(u)
        return _diam_from_dist(_bfs(n, adj), n)

    cur = []
    for _ in range(60):
        d = diam(cur)
        if d <= 4:
            break
        best_e, best_d = None, d
        for e in cands:
            if e in cur:
                continue
            td = diam(cur + [e])
            if td >= 0 and td < best_d:
                best_d, best_e = td, e
        if best_e is None:
            break
        cur.append(best_e)
    # single-removal minimization
    improved = True
    while improved:
        improved = False
        for i in range(len(cur)):
            trial = cur[:i] + cur[i + 1:]
            if _diam_from_dist(_bfs(n, _base_adj(n, base_edges) if False else _add(cur, n, base_edges)), n) <= 4:
                cur = trial; improved = True; break
    return cur


def _add(added, n, base_edges):
    adj = _base_adj(n, base_edges)
    for u, v in added:
        adj[u].add(v); adj[v].add(u)
    return adj


def construct():
    t0 = time.time()
    DEADLINE = 82.0
    n = P['n']
    base_edges = [tuple(e) for e in P['edges']]
    base_set = set()
    for u, v in base_edges:
        base_set.add((u, v)); base_set.add((v, u))

    adj0 = _base_adj(n, base_edges)
    dist0 = _bfs(n, adj0)
    even = [i for i in range(n) if i % 2 == 0]
    odd = [i for i in range(n) if i % 2 == 1]
    cands = [(u, v) for u in even for v in odd if (u, v) not in base_set]
    cm = len(cands)

    # Precompute triangle-free compatibility between candidate edges.
    # Two edges (a,b) and (c,d) are compatible iff they are vertex-disjoint AND
    # no endpoint is adjacent to an endpoint of the other (in base graph).
    comp = [None] * cm
    for i in range(cm):
        a, b = cands[i]
        row = [True] * cm
        row[i] = False
        na = adj0[a] | adj0[b]
        for j in range(cm):
            if j == i:
                continue
            c, d = cands[j]
            if c in (a, b) or d in (a, b):
                row[j] = False
            elif c in na or d in na:
                row[j] = False
        comp[i] = row

    # Precompute per-vertex candidate indices.
    vidx = {}
    for idx, (u, v) in enumerate(cands):
        vidx.setdefault(u, []).append(idx)
        vidx.setdefault(v, []).append(idx)

    # Constructive heuristic -> upper bound.
    heuristic = _constructive(n, base_edges)
    ub = len(heuristic)
    best_sol = list(heuristic)

    def time_left():
        return time.time() < DEADLINE

    # Branch-and-bound.
    # State: chosen candidate indices (list), dist matrix (list of lists),
    #        per-vertex chosen edge count (int list), allowed mask per index.
    # We build incrementally and prune.

    import sys
    sys.setrecursionlimit(10000)

    # Use iterative deepening on the target size to keep search focused.
    # Try target = ub, ub-1, ... down to 1; first feasible target gives optimum.
    # But B&B with global ub already prunes; we do a single B&B run minimizing.

    chosen = []
    chosen_count = [0] * n
    distm = [list(row) for row in dist0]
    allowed = [True] * cm

    def apply(i):
        u, v = cands[i]
        # update dist
        du, dv = distm[u], distm[v]
        for x in range(n):
            nd = du[x] + 1
            if nd < dv[x]:
                dv[x] = nd
            nd2 = dv[x] + 1
            if nd2 < du[x]:
                du[x] = nd2
        chosen_count[u] += 1
        chosen_count[v] += 1
        for j in range(cm):
            if allowed[j] and not comp[i][j]:
                allowed[j] = False

    def unapply(i):
        u, v = cands[i]
        # rebuild dist from scratch is simplest & safe (n small)
        adj = _base_adj(n, base_edges)
        for k in chosen:
            a, b = cands[k]
            adj[a].add(b); adj[b].add(a)
        distm[:] = [list(r) for r in _bfs(n, adj)]
        chosen_count[u] -= 1
        chosen_count[v] -= 1
        # recompute allowed from scratch
        for j in range(cm):
            ok = True
            for k in chosen:
                if k == j:
                    continue
                if not comp[k][j]:
                    ok = False
                    break
            allowed[j] = ok

    def lower_bound():
        # diameter-4 lower bound: for farthest pair, need >= ceil((d-4)/2) more edges
        best = 0
        for i in range(n):
            row = distm[i]
            for j in range(i + 1, n):
                d = row[j]
                if d < 0:
                    return 10 ** 9
                need = (d - 3) // 2  # each new edge can reduce by at most 2
                if need < 0:
                    need = 0
                if need > best:
                    best = need
        return best

    def pick_var():
        # vertex with most unmet (dist>4) pairs, among vertices with available candidates
        best_v, best_score = None, -1
        for v in range(n):
            cnt = chosen_count[v]
            if cnt > 1:
                continue
            # count pairs (v, w) with dist>4 that are not already "covered"
            row = distm[v]
            score = 0
            for w in range(n):
                if w == v:
                    continue
                if row[w] > 4:
                    score += 1
            # prefer vertices that have at least one allowed candidate
            has_cand = False
            for idx in vidx.get(v, ()):
                if allowed[idx]:
                    has_cand = True
                    break
            if not has_cand:
                if score > 0:
                    return None  # dead end: vertex has unmet pairs but no candidate
            if score > best_score:
                best_score = score
                best_v = v
        return best_v

    def solve():
        nonlocal ub, best_sol
        if not time_left():
            return
        if len(chosen) >= ub:
            return
        # lower bound prune
        lb = lower_bound()
        if lb + len(chosen) >= ub:
            return
        # check if already diameter <= 4
        dmax = 0
        for i in range(n):
            row = distm[i]
            for j in range(i + 1, n):
                if row[j] < 0:
                    return
                if row[j] > dmax:
                    dmax = row[j]
        if dmax <= 4:
            if len(chosen) < ub:
                ub = len(chosen)
                best_sol = [cands[k] for k in chosen]
            return
        # pick variable
        v = pick_var()
        if v is None:
            return
        # order candidates for v by how much they reduce dmax (heuristic)
        cidxs = [idx for idx in vidx.get(v, ()) if allowed[idx]]
        if not cidxs:
            return
        # sort: prefer edges whose endpoints are far from many vertices
        def score(idx):
            a, b = cands[idx]
            s = 0
            for x in range(n):
                s += distm[a][x] + distm[b][x]
            return -s
        cidxs.sort(key=score)
        for idx in cidxs:
            if not time_left():
                return
            if not allowed[idx]:
                continue
            # triangle-free check already in allowed
            chosen.append(idx)
            apply(idx)
            solve()
            unapply(idx)
            chosen.pop()
            if len(chosen) + 1 >= ub:
                # even adding more won't help beyond current; but we still explore
                pass

    # Seed the search with the heuristic edges to give a good starting point
    # (so ub is already tight). Then B&B tries to beat it.
    # Map heuristic edges back to candidate indices.
    edge_to_idx = {e: i for i, e in enumerate(cands)}
    heuristic_indices = []
    for e in heuristic:
        e2 = (e[0], e[1])
        if e2 in edge_to_idx:
            heuristic_indices.append(edge_to_idx[e2])
        else:
            e3 = (e[1], e[0])
            if e3 in edge_to_idx:
                heuristic_indices.append(edge_to_idx[e3])
    # Ensure heuristic is triangle-free compatible (it should be).
    ok = True
    for a in range(len(heuristic_indices)):
        for b in range(a + 1, len(heuristic_indices)):
            if not comp[heuristic_indices[a]][heuristic_indices[b]]:
                ok = False
    if ok:
        for idx in heuristic_indices:
            chosen.append(idx)
            apply(idx)
        solve()
        # pop them
        for _ in range(len(heuristic_indices)):
            unapply(chosen[-1])
            chosen.pop()

    # Final validity check; fall back to heuristic if needed.
    sol = best_sol
    # verify sol
    def valid(sol):
        for u, v in sol:
            if u == v:
                return False
            if u % 2 == v % 2:
                return False
            if (u, v) in base_set or (v, u) in base_set:
                return False
        adj = _base_adj(n, base_edges)
        for u, v in sol:
            adj[u].add(v); adj[v].add(u)
        d = _diam_from_dist(_bfs(n, adj), n)
        return d >= 0 and d <= 4
    if not valid(sol):
        sol = heuristic
    if not valid(sol):
        # last resort: connect 12 to all odd non-adjacent, 11 to all even non-adjacent
        sol = []
        for v in odd:
            if (12, v) not in base_set:
                sol.append((12, v))
        for v in even:
            if (11, v) not in base_set:
                sol.append((11, v))
        if not valid(sol):
            sol = []

    # Type-safe output
    result = []
    seen = set()
    base_keys = set()
    for a, b in base_set:
        if a < b:
            base_keys.add((a, b))
    for e in sol:
        u, v = int(e[0]), int(e[1])
        key = (min(u, v), max(u, v))
        if key in seen or key in base_keys:
            continue
        seen.add(key)
        result.append([u, v])
    return result
# EVOLVE-BLOCK-END
