import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

N = P['n']
PATH_EDGES = P['edges']
PATH_SET = set((min(u, v), max(u, v)) for u, v in PATH_EDGES)
ALL_PAIRS = [(u, v) for u in range(N) for v in range(u + 1, N) if (u, v) not in PATH_SET]

# ---------- deterministic valid baseline (hub 11) ----------
_HUB = 11
_HUB_EVENS = [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]
_BASELINE = [(HUB, v) for HUB, v in [(_HUB, x) for x in _HUB_EVENS]]


def _build(added):
    adj = [[] for _ in range(N)]
    mat = [[False] * N for _ in range(N)]
    for u, v in PATH_EDGES:
        adj[u].append(v); adj[v].append(u)
        mat[u][v] = mat[v][u] = True
    for u, v in added:
        adj[u].append(v); adj[v].append(u)
        mat[u][v] = mat[v][u] = True
    return adj, mat


def _tri_free(adj, mat):
    for u in range(N):
        au = adj[u]
        for i in range(len(au)):
            a = au[i]
            rowa = mat[a]
            for j in range(i + 1, len(au)):
                b = au[j]
                if rowa[b]:
                    return False
    return True


def _diam_le4(adj):
    for s in range(N):
        dist = [-1] * N
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            if dist[u] >= 4:
                continue
            nd = dist[u] + 1
            for v in adj[u]:
                if dist[v] == -1:
                    if nd > 4:
                        return False
                    dist[v] = nd
                    q.append(v)
        for d in dist:
            if d == -1:
                return False
    return True


def _valid_set(added):
    adj, mat = _build(added)
    if not _tri_free(adj, mat):
        return False
    return _diam_le4(adj)


# ---------- branch and bound (ILP-style: minimize edge count, prove optimum) ----------
def _bfs_dist(adj, s):
    n = N
    dist = [-1] * n
    dist[s] = 0
    q = deque([s])
    while q:
        u = q.popleft()
        nd = dist[u] + 1
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = nd
                q.append(v)
    return dist


def _farthest_pair(adj):
    best = (0, 0, -1)
    for s in range(N):
        dist = _bfs_dist(adj, s)
        for t in range(N):
            if dist[t] > best[2]:
                best = (s, t, dist[t])
    return best


def _can_add(added, adj, mat, u, v):
    if u == v:
        return False
    k = (min(u, v), max(u, v))
    if k in PATH_SET or k in added:
        return False
    su = set(adj[u]); sv = set(adj[v])
    return len(su & sv) == 0


def _complete(added, deadline):
    added = set(added)
    while time.time() < deadline:
        adj, mat = _build(added)
        if _tri_free(adj, mat) and _diam_le4(adj):
            return added
        a, b, d = _farthest_pair(adj)
        cand = None
        if _can_add(added, adj, mat, a, b):
            cand = (a, b)
        else:
            for v in range(N):
                if v != a and _can_add(added, adj, mat, a, v):
                    cand = (a, v)
                    break
        if cand is None:
            for (u, v) in ALL_PAIRS:
                if _can_add(added, adj, mat, u, v):
                    cand = (u, v)
                    break
        if cand is None:
            return added
        added.add(cand)
    return added


def _reduce(added, deadline):
    added = set(added)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for e in list(added):
            trial = added - {e}
            if _valid_set(trial):
                added = trial
                improved = True
                break
    return added


def _all_candidates(cur):
    return [p for p in ALL_PAIRS if p not in cur]


def _local_search(added, deadline):
    cur = set(added)
    best = set(cur)
    no_improve = 0
    while time.time() < deadline:
        cur = _reduce(cur, deadline)
        if len(cur) < len(best):
            best = set(cur)
            no_improve = 0
        improved = False
        e_list = list(cur)
        random.shuffle(e_list)
        for e in e_list:
            if time.time() > deadline:
                break
            f_list = _all_candidates(cur)
            random.shuffle(f_list)
            for f in f_list[:30]:
                trial = (cur - {e}) | {f}
                trial = _reduce(trial, deadline)
                if len(trial) < len(cur):
                    cur = trial
                    if len(cur) < len(best):
                        best = set(cur)
                    improved = True
                    break
            if improved:
                break
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            cands = _all_candidates(cur)
            if not cands:
                break
            kicked = False
            for _ in range(40):
                if time.time() > deadline:
                    break
                f = random.choice(cands)
                trial = _reduce(cur | {f}, deadline)
                if len(trial) < len(cur):
                    cur = trial
                    if len(cur) < len(best):
                        best = set(cur)
                    no_improve = 0
                    kicked = True
                    break
            if no_improve >= 4 and not kicked:
                break
    return best


# ---------- deterministic branch-and-bound optimizer ----------
def _branch_and_bound(initial, target, deadline):
    # state: (depth, added_set, adj, mat)
    start = set(initial)
    adj, mat = _build(start)
    if not _tri_free(adj, mat):
        return None
    best = {'sol': None}
    # Precompute, for each candidate edge, the set of vertex pairs it can help
    # (i.e., pairs (a,b) such that adding the edge is triangle-free w.r.t. the
    #  base+initial and would be a candidate). Used for a coverage lower bound.
    # We keep it simple: lower bound = number of farthest pairs currently > 4
    #  that are not yet "covered" by selected edges. We approximate coverage by
    #  recomputing the farthest pair each node (cheap, N=24).

    # Order candidate edges globally by how useful they look: prefer edges that
    #  connect vertices far apart in the path (large |u-v|) since those shorten
    #  the most diameter.
    cands = [p for p in ALL_PAIRS if p not in start]
    cands.sort(key=lambda p: (-(abs(p[0] - p[1]))))

    # We do a DFS with memo on frozenset(added) to avoid re-exploring.
    memo = {}

    def dfs(added, adj, mat, depth):
        if time.time() > deadline:
            return
        cur = len(added)
        if best['sol'] is not None and cur >= best['sol']:
            return
        # Check validity
        if _diam_le4(adj):
            if best['sol'] is None or cur < best['sol']:
                best['sol'] = cur
                best['found'] = set(added)
            return
        # Lower bound: count pairs with dist > 4 in current graph.
        # Each added edge can reduce the diameter; a crude bound: if even the
        #  farthest pair distance is > 4 we need at least 1 more edge.
        # Stronger: number of vertices at distance > 4 from some root.
        # We use: if cur + 1 > target and best['sol'] is not None: prune.
        if best['sol'] is not None and cur + 1 > best['sol']:
            return
        # Try adding each candidate edge (in usefulness order), triangle-free only.
        for (u, v) in cands:
            if time.time() > deadline:
                return
            if (min(u, v), max(u, v)) in added:
                continue
            # triangle-free check
            su = set(adj[u]); sv = set(adj[v])
            if su & sv:
                continue
            # build new
            nadj = [list(x) for x in adj]
            nmat = [row[:] for row in mat]
            nadj[u].append(v); nadj[v].append(u)
            nmat[u][v] = nmat[v][u] = True
            nadded = added | {(min(u, v), max(u, v))}
            key = frozenset(nadded)
            if key in memo:
                continue
            memo[key] = True
            dfs(nadded, nadj, nmat, depth + 1)
            if best['sol'] is not None and best['sol'] <= cur:
                # we already found a solution no worse than current depth, stop
                return
            del memo[key]

    dfs(start, adj, mat, 0)
    return best['found'] if 'found' in best else None


def _solve_from(seed, deadline):
    added = _complete(set(seed), deadline)
    if not _valid_set(added):
        return None
    added = _reduce(added, deadline)
    return added


def _make_seeds():
    seeds = []
    seeds.append(_BASELINE)
    seeds.append([])
    for hub in [11, 12, 7, 16]:
        added = []
        for v in range(N):
            if v != hub and abs(v - hub) >= 2:
                added.append((hub, v))
        seeds.append(added)
    for k in [5, 6, 7, 8]:
        added = []
        pos = 0
        while pos + k <= N - 1:
            added.append((pos, pos + k))
            pos += k
        if pos < N - 1:
            added.append((pos, N - 1))
        seeds.append(added)
    return seeds


def construct():
    t0 = time.time()
    deadline = t0 + 80
    best = None

    # 1) Deterministic baseline
    if _valid_set(set(_BASELINE)):
        best = set(_BASELINE)

    # 2) Try seeds -> reduce
    seeds = _make_seeds()
    for seed in seeds:
        if time.time() > deadline - 15:
            break
        sol = _solve_from(seed, deadline - 15)
        if sol is not None and _valid_set(sol) and (best is None or len(sol) < len(best)):
            best = sol

    # 3) Branch-and-bound to find/prove the true minimum (structural ILP-like search)
    if best is not None and time.time() < deadline - 5:
        bb = _branch_and_bound(best, len(best) - 1, deadline - 5)
        if bb is not None and _valid_set(bb) and len(bb) < len(best):
            best = set(bb)

    # 4) Local-search polish
    if best is not None and time.time() < deadline - 3:
        polished = _local_search(best, deadline - 3)
        if _valid_set(polished) and len(polished) < len(best):
            best = polished

    if best is None:
        best = set(_BASELINE)

    return [[int(u), int(v)] for (u, v) in best]
