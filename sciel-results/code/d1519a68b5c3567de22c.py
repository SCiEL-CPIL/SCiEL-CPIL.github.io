"""Triangle-free diameter-4 augmentation of P_24 via bipartite set cover.

Key structural facts:
  * Base graph is the path P_24 (vertices 0..23).
  * Adding edges between even and odd vertices keeps the graph bipartite,
    hence automatically triangle-free.  So the ONLY nontrivial constraint is
    diameter <= 4.  (We still verify triangle-freedom defensively.)
  * Every pair with path-distance > 4 must be covered by at least one added
    edge.  Each added edge covers a fixed set of such pairs.  => set cover.
"""
import time
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
                        [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14],
                        [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20],
                        [20, 21], [21, 22], [22, 23]], 'diameter': 4}

N = P['n']
BASE = P['edges']
D = 4
WALL = 90.0
DEADLINE = time.time() + WALL - 4.0   # safety margin before the 90s kill

# ---- static precomputation (path distances, bad pairs, edge coverage) ----
_path = [[0] * N for _ in range(N)]
for _i in range(N):
    for _j in range(N):
        _path[_i][_j] = abs(_i - _j)

BAD = [(i, j) for i in range(N) for j in range(i + 1, N) if _path[i][j] > D]
BAD_SET = set(BAD)

CANDS = [(u, v) for u in range(N) for v in range(u + 1, N)
         if (u, v) not in {(a, b) for a, b in BASE}]

# coverage[e][b] = True iff adding edge e brings bad pair b within distance D
# (computed with the exact shortest-path formula: the only new path uses e once)
_coverage = []
for e in CANDS:
    u, v = e
    cov = [False] * len(BAD)
    for bi, (i, j) in enumerate(BAD):
        nd = _path[i][j]
        a = _path[i][u] + 1 + _path[v][j]
        if a < nd:
            nd = a
        b = _path[i][v] + 1 + _path[u][j]
        if b < nd:
            nd = b
        cov[bi] = (nd <= D)
    _coverage.append(cov)

# ---- fast graph distance helpers (list adjacency, BFS) ----
def _build_adj(added):
    adj = [[] for _ in range(N)]
    for u, v in BASE:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _bfs(s, adj):
    dist = [-1] * N
    dist[s] = 0
    dq = [s]
    while dq:
        nq = []
        for u in dq:
            du = dist[u] + 1
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = du
                    nq.append(v)
        dq = nq
    return dist


def _diam_ok(adj):
    for s in range(N):
        d = _bfs(s, adj)
        if any(x < 0 or x > D for x in d):
            return False
    return True


def _tri_ok(adj):
    for u in range(N):
        su = set(adj[u])
        for v in adj[u]:
            if v <= u:
                continue
            if su & set(adj[v]):
                return False
    return True


def _valid(added):
    adj = _build_adj(added)
    return _tri_ok(adj) and _diam_ok(adj)


def _bad_from_dist(adj):
    bad = set()
    for i in range(N):
        d = _bfs(i, adj)
        for j in range(i + 1, N):
            if d[j] > D:
                bad.add((i, j))
    return bad


def _covered_bad(added):
    """Set of bad pairs (i,j) whose distance is brought <= D by at least one
    edge in `added`.  Derived exactly (not from the path-only precompute)."""
    adj = _build_adj(added)
    bad = _bad_from_dist(adj)
    if not bad:
        return set()
    # A bad pair is covered iff removing some added edge makes it worse / unreachable.
    # Cheaper: pair is covered iff there exists edge e=(u,v) in added such that
    # the best path through e is <= D while the path-only distance is > D.
    covered = set()
    for (i, j) in bad:
        base_d = _path[i][j]
        if base_d <= D:
            continue
        for (u, v) in added:
            a = _path[i][u] + 1 + _path[v][j]
            b = _path[i][v] + 1 + _path[u][j]
            if min(a, b) <= D:
                covered.add((i, j))
                break
    return covered


def _greedy_set_cover(rng=None):
    """Greedy set cover over bad pairs.  Incremental uncovered counter.
    Returns (added, uncovered_count)."""
    # uncovered[b] = number of candidate edges (not yet chosen) that cover bad pair b
    uncovered = [0] * len(BAD)
    for e in range(len(CANDS)):
        cov = _coverage[e]
        for b in range(len(BAD)):
            if cov[b]:
                uncovered[b] += 1
    chosen = []
    chosen_set = set()
    while True:
        rem = 0
        for b in range(len(BAD)):
            if uncovered[b] > 0:
                rem += 1
        if rem == 0:
            break
        best_e = -1
        best_key = None
        for e in range(len(CANDS)):
            if e in chosen_set:
                continue
            cov = _coverage[e]
            cnt = 0
            for b in range(len(BAD)):
                if cov[b] and uncovered[b] > 0:
                    cnt += 1
            if cnt <= 0:
                continue
            if best_key is None or cnt > best_key[0]:
                best_key = (cnt, e)
                best_e = e
        if best_e < 0:
            break
        cov = _coverage[best_e]
        for b in range(len(BAD)):
            if cov[b]:
                uncovered[b] -= 1
        chosen_set.add(best_e)
        chosen.append(CANDS[best_e])
    rem = sum(1 for b in range(len(BAD)) if uncovered[b] > 0)
    return chosen, rem


def _local_search(added, deadline):
    """Removal + 1-for-1 swap, keeping validity.  Returns best valid list."""
    cur = [list(e) for e in added]
    if not _valid(cur):
        return cur
    best = list(cur)
    best_len = len(cur)

    # ---- removal pass ----
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(cur)):
            cand = cur[:idx] + cur[idx + 1:]
            if _valid(cand):
                cur = cand
                best = list(cur)
                best_len = len(cur)
                changed = True
                break

    # ---- swap pass (1-for-1) ----
    while time.time() < deadline:
        improved = False
        cur_set = {tuple(e) for e in cur}
        for idx in range(len(cur)):
            if time.time() > deadline:
                break
            for c in CANDS:
                if c in cur_set:
                    continue
                cand = cur[:idx] + cur[idx + 1:] + [list(c)]
                if _valid(cand):
                    cur = cand
                    best = list(cur)
                    best_len = len(cur)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def construct():
    deadline = DEADLINE
    best = None
    best_len = 10**9

    # deterministic greedy
    g, rem = _greedy_set_cover()
    if _valid(g):
        best = g
        best_len = len(g)

    # local search on the greedy result
    if best is not None and time.time() < deadline - 5:
        ls = _local_search(best, deadline - 5)
        if _valid(ls) and len(ls) <= best_len:
            best = ls
            best_len = len(ls)

    # randomized restarts: shuffle candidate order, greedy, local search
    if time.time() < deadline - 6:
        rng = random.Random(12345)
        order = list(range(len(CANDS)))
        while time.time() < deadline - 6:
            rng.shuffle(order)
            # greedy with this fixed order but still max-coverage tie-break
            chosen = []
            chosen_set = set()
            uncovered = [0] * len(BAD)
            for e in order:
                cov = _coverage[e]
                for b in range(len(BAD)):
                    if cov[b]:
                        uncovered[b] += 1
            while True:
                rem = sum(1 for b in range(len(BAD)) if uncovered[b] > 0)
                if rem == 0:
                    break
                best_e = -1
                best_cnt = -1
                for e in order:
                    if e in chosen_set:
                        continue
                    cov = _coverage[e]
                    cnt = 0
                    for b in range(len(BAD)):
                        if cov[b] and uncovered[b] > 0:
                            cnt += 1
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_e = e
                if best_e < 0 or best_cnt <= 0:
                    break
                cov = _coverage[best_e]
                for b in range(len(BAD)):
                    if cov[b]:
                        uncovered[b] -= 1
                chosen_set.add(best_e)
                chosen.append(CANDS[best_e])
            if _valid(chosen) and len(chosen) < best_len:
                best = chosen
                best_len = len(chosen)
                if time.time() < deadline - 6:
                    ls = _local_search(best, deadline - 6)
                    if _valid(ls) and len(ls) <= best_len:
                        best = ls
                        best_len = len(ls)

    if best is None:
        # absolute fallback: exhaustive add of all cross-parity edges
        best = [list(e) for e in CANDS]
        if not _valid(best):
            best = []

    out = [[int(u), int(v)] for (u, v) in best]
    return out
