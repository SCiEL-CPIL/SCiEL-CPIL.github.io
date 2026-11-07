import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
                        [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
                        [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]],
     'diameter': 4}

N = P['n']
PATH_EDGES = P['edges']
PATH_SET = set((min(u, v), max(u, v)) for u, v in PATH_EDGES)
ALL_PAIRS = [(u, v) for u in range(N) for v in range(u + 1, N) if (u, v) not in PATH_SET]

# ---- base graph ----
BASE_ADJ = [[] for _ in range(N)]
BASE_MAT = [[False] * N for _ in range(N)]
for u, v in PATH_EDGES:
    BASE_ADJ[u].append(v); BASE_ADJ[v].append(u)
    BASE_MAT[u][v] = BASE_MAT[v][u] = True

# ---- bipartition (path is bipartite): evens / odds ----
PART = [i % 2 for i in range(N)]
EVEN = [i for i in range(N) if PART[i] == 0]
ODD = [i for i in range(N) if PART[i] == 1]
# bipartite candidates: all cross edges except the path edges
BIP_CANDS = [(u, v) for (u, v) in ALL_PAIRS if PART[u] != PART[v]]


def _bipartite(adj):
    col = [-1] * N
    for s in range(N):
        if col[s] != -1:
            continue
        col[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if col[v] == -1:
                    col[v] = col[u] ^ 1
                    q.append(v)
                elif col[v] == col[u]:
                    return False
    return True


def _build(added):
    adj = [list(x) for x in BASE_ADJ]
    mat = [row[:] for row in BASE_MAT]
    for u, v in added:
        adj[u].append(v); adj[v].append(u)
        mat[u][v] = mat[v][u] = True
    return adj, mat


def _dist(adj, s):
    n = N
    d = [-1] * n
    d[s] = 0
    q = deque([s])
    while q:
        u = q.popleft()
        nd = d[u] + 1
        for v in adj[u]:
            if d[v] == -1:
                d[v] = nd
                q.append(v)
    return d


def _diam_le4(adj):
    for s in range(N):
        d = _dist(adj, s)
        for x in d:
            if x == -1 or x > 4:
                return False
    return True


def _valid(added):
    adj, mat = _build(added)
    if not _bipartite(adj):
        return False
    return _diam_le4(adj)


def _coverage(added, e):
    """pairs whose distance becomes <=4 when adding e (triangle-free assumed)."""
    adj, _ = _build(added)
    d0 = _dist(adj, e[0])
    adj2 = [list(x) for x in adj]
    adj2[e[0]].append(e[1]); adj2[e[1]].append(e[0])
    d1 = _dist(adj2, e[0])
    cov = set()
    for v in range(N):
        if d0[v] > 4 and d1[v] <= 4:
            cov.add((min(e[0], v), max(e[0], v)))
    return cov


def _greedy_cover(added):
    added = set(added)
    for _ in range(500):
        adj, _ = _build(added)
        d0 = _dist(adj, 0)
        if all(d0[v] <= 4 for v in range(N)):
            break
        best_e, best_cov = None, set()
        for e in BIP_CANDS:
            if e in added:
                continue
            if BASE_MAT[e[0]][e[1]]:
                continue
            # triangle-free check (bipartite)
            if any(BASE_MAT[e[0]][w] and BASE_MAT[e[1]][w] for w in range(N)):
                continue
            cov = _coverage(added, e)
            if len(cov) > len(best_cov):
                best_cov, best_e = cov, e
        if best_e is None:
            break
        added.add(best_e)
    return added


# ---- exact branch-and-bound set cover ----
def _bipartite_ok(added, u, v):
    return not any(BASE_MAT[u][w] and BASE_MAT[v][w] for w in range(N))


def _solve_setcover(added, budget, deadline):
    adj, _ = _build(added)
    d0 = _dist(adj, 0)
    uncovered = set()
    for v in range(N):
        if d0[v] > 4:
            uncovered.add((min(0, v), max(0, v)))
    if not uncovered:
        return set(added)
    cands = [e for e in BIP_CANDS if e not in added and _bipartite_ok(added, e[0], e[1])]
    covs = {}
    for e in cands:
        c = _coverage(added, e)
        if c:
            covs[e] = c & uncovered
    # order candidates by coverage size desc
    order = sorted(covs, key=lambda e: (-len(covs[e]), e))
    best = {'sol': None}
    n = len(order)
    idx_of = {e: i for i, e in enumerate(order)}
    idx_of = {e: i for i, e in enumerate(order)}

    def dfs(added, uncovered, start, used):
        if time.time() > deadline:
            return
        if best['sol'] is not None and used >= best['sol']:
            return
        if not uncovered:
            best['sol'] = used
            best['found'] = set(added)
            return
        # lower bound: need at least ceil(|uncovered| / max_cov)
        if order:
            maxcov = 0
            for i in range(start, n):
                if covs[order[i]] & uncovered:
                    lc = len(covs[order[i]] & uncovered)
                    if lc > maxcov:
                        maxcov = lc
            if maxcov == 0:
                return
            lb = (len(uncovered) + maxcov - 1) // maxcov
            if used + lb >= (best['sol'] if best['sol'] is not None else budget):
                return
        for i in range(start, n):
            if time.time() > deadline:
                return
            e = order[i]
            c = covs[e] & uncovered
            if not c:
                continue
            added.add(e)
            dfs(added, uncovered - c, i + 1, used + 1)
            added.remove(e)
            if best['sol'] is not None and best['sol'] <= used:
                return

    dfs(set(added), uncovered, 0, 0)
    return best['found'] if 'found' in best else None


def _construct():
    t0 = time.time()
    deadline = t0 + 82
    best = None

    # greedy warm start
    g = _greedy_cover(set())
    if _valid(g) and (best is None or len(g) < len(best)):
        best = set(g)

    # iterative deepening exact search to prove optimality
    lo = 0
    if best is not None:
        lo = max(0, len(best) - 1)
    hi = len(best) if best is not None else 12
    for budget in range(lo, hi + 1):
        if time.time() > deadline - 3:
            break
        sol = _solve_setcover(set(), budget, deadline - 3)
        if sol is not None and _valid(sol) and len(sol) <= budget:
            best = set(sol)
            break
    if best is None:
        # fallback: hub star
        best = set()
        for v in EVEN:
            if v != 11:
                best.add((11, v))
        if not _valid(best):
            best = set(g)
    return [[int(u), int(v)] for (u, v) in best]


def construct():
    res = _construct()
    return [[int(u), int(v)] for (u, v) in res]
