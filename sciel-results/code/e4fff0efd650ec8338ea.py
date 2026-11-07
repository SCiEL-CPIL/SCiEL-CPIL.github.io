import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin under 90s limit

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    degrees = [len(adj[i]) for i in range(n)]

    # Precompute all-pairs k-shortest simple paths (interior-disjointness checked later).
    # For each ordered pair (s,t), store a list of candidate paths (as tuples of ints),
    # shortest first, with distinct interiors.
    KPATH = 8
    def kshortest(s, t, kmax):
        if s == t:
            return []
        seen = set()
        out = []
        # BFS by distance to get shortest paths; collect up to kmax distinct paths.
        # Use parent tracking to enumerate simple shortest paths.
        # Build BFS tree with parents (list of parents per node) to enumerate shortest paths.
        dist = [-1] * n
        parents = [[] for _ in range(n)]
        q = deque()
        q.append(s)
        dist[s] = 0
        while q:
            cur = q.popleft()
            for nb in adj[cur]:
                if dist[nb] == -1:
                    dist[nb] = dist[cur] + 1
                    parents[nb].append(cur)
                    q.append(nb)
        if dist[t] == -1:
            return []
        # Enumerate shortest paths from t back to s via parent DAG.
        def backtrack(node, path):
            if len(out) >= kmax:
                return
            if node == s:
                p = tuple(reversed(path))
                interior = p[1:-1]
                key = frozenset(interior)
                if key not in seen:
                    seen.add(key)
                    out.append(p)
                return
            for par in parents[node]:
                if dist[par] == dist[node] - 1:
                    backtrack(par, path + [par])
        backtrack(t, [t])
        return out

    path_cache = {}
    def get_paths(a, b):
        key = (a, b)
        if key not in path_cache:
            path_cache[key] = kshortest(a, b, KPATH)
        return path_cache[key]

    # Check a branch set: try to assign pairwise interior-disjoint paths.
    def try_branches(branches):
        branches_set = set(branches)
        pairs = list(combinations(branches, 2))
        # Precompute candidate paths for each pair.
        cand = []
        for (u, v) in pairs:
            ps = get_paths(u, v)
            if not ps:
                return None
            cand.append(ps)
        m = len(pairs)
        used = [0] * n  # used[v] = 1 if interior vertex v is used
        assign = [None] * m

        def rec(i):
            if time.time() > deadline:
                return False
            if i == m:
                return True
            for p in cand[i]:
                ok = True
                for v in p[1:-1]:
                    if used[v]:
                        ok = False
                        break
                if not ok:
                    continue
                for v in p[1:-1]:
                    used[v] = 1
                assign[i] = p
                if rec(i + 1):
                    return True
                assign[i] = None
                for v in p[1:-1]:
                    used[v] = 0
            return False

        if rec(0):
            return [list(assign[i]) for i in range(m)]
        return None

    # Maximum clique of adjacency graph restricted to a candidate pool (Bron-Kerbosch w/ pivot).
    def max_clique(pool):
        pool_set = set(pool)
        best = []
        def bk(R, P, X):
            nonlocal best
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            if P or X:
                pivot = max(list(P) + list(X), key=lambda v: len(adj[v] & P))
                cands = [v for v in P if v not in adj[pivot]]
            else:
                cands = list(P)
            for v in cands:
                bk(R + [v], P & adj[v], X & adj[v])
                P.discard(v)
                X.add(v)
        bk([], pool_set, set())
        return best

    # Enumerate all maximal cliques (up to a cap) in a pool, return list of cliques.
    def all_maximal_cliques(pool, cap=200):
        pool_set = set(pool)
        found = []
        def bk(R, P, X):
            nonlocal found
            if time.time() > deadline:
                return
            if not P and not X:
                if R:
                    found.append(list(R))
                return
            if len(found) >= cap:
                return
            if P or X:
                pivot = max(list(P) + list(X), key=lambda v: len(adj[v] & P))
                cands = [v for v in P if v not in adj[pivot]]
            else:
                cands = list(P)
            for v in cands:
                bk(R + [v], P & adj[v], X & adj[v])
                P.discard(v)
                X.add(v)
        bk([], pool_set, set())
        return found

    best = None
    best_size = -1

    def consider(branches):
        nonlocal best, best_size
        if time.time() > deadline:
            return
        if len(branches) <= best_size:
            return
        paths = try_branches(list(branches))
        if paths is not None:
            best = {
                'branches': [int(b) for b in branches],
                'paths': [list(map(int, p)) for p in paths],
            }
            best_size = len(branches)

    # Candidate pools: high-degree vertices, plus a few random restarts.
    order = sorted(range(n), key=lambda x: degrees[x], reverse=True)

    # Restart 0: top 14 by degree
    base_pool = order[:14]
    cliques = all_maximal_cliques(base_pool, cap=400)
    # Sort cliques by size desc
    cliques.sort(key=lambda c: len(c), reverse=True)
    for c in cliques:
        consider(c)
        if best_size >= 10:
            break

    # Additional random restarts to explore other branch sets
    random.seed(12345)
    for restart in range(1, 40):
        if time.time() > deadline:
            break
        # Pool: top 10 by degree + 6 random
        pool = set(order[:10])
        pool.update(random.sample(range(n), 6))
        cliques = all_maximal_cliques(pool, cap=200)
        cliques.sort(key=lambda c: len(c), reverse=True)
        for c in cliques:
            consider(c)
            if best_size >= 10:
                break
        if best_size >= 10:
            break

    if best is not None:
        return best

    # Fallback: max clique of whole graph (trivial subdivision)
    mc = max_clique(range(n))
    if len(mc) >= 2:
        branches = [int(v) for v in mc]
        paths = [[int(u), int(v)] for u, v in combinations(branches, 2)]
        return {'branches': branches, 'paths': paths}

    # Absolute fallback
    u, v = edges[0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
