"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin vs 90s kill

    n = P['n']
    edges = P['edges']

    def check_time():
        return time.time() < deadline

    # ---- adjacency ----
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # ---- precompute ALL interior-disjoint branch-avoiding simple paths between each ordered pair ----
    # paths[u][v] = list of (interior_mask, path_list)
    paths = [[[] for _ in range(n)] for _ in range(n)]

    def enumerate_paths(u, v, forbidden, path, used, out):
        # forbidden: set of vertices that must not be used (branches other than u,v)
        for w in adj[path[-1]]:
            if w == u:
                # found a simple path u..w(=u); interiors = path[1:-1]
                out.append((used, list(path)))
                continue
            if w == v:
                # path u..v
                out.append((used, list(path)))
                continue
            if w in used or w in forbidden:
                continue
            used.add(w)
            path.append(w)
            enumerate_paths(u, v, forbidden, path, used, out)
            path.pop()
            used.remove(w)

    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            out = []
            # initial step
            for w in adj[u]:
                if w == v:
                    out.append((0, [u, v]))
                    continue
                if w == u:
                    continue
                used = {w}
                path = [u, w]
                enumerate_paths(u, v, set(), path, used, out)
            # dedupe by (interior_mask, path)
            seen = set()
            dedup = []
            for mask, p in out:
                key = (mask, tuple(p))
                if key in seen:
                    continue
                seen.add(key)
                dedup.append((mask, p))
            # keep only a bounded, diverse subset: shortest first, then spread masks
            dedup.sort(key=lambda t: (len(t[1]), t[0]))
            paths[u][v] = dedup

    # ---- feasibility: given branch set, can we pick one path per pair with disjoint interiors? ----
    # We solve a set-packing: for each pair choose a path; interiors (bitmask) must be disjoint.
    # Use DP over pairs is exponential; instead use randomized greedy + local search with the
    # precomputed path lists (much faster than re-running BFS each time).

    def try_assign(branches, seed=0):
        """Return dict pair->path or None. Uses randomized greedy over precomputed paths."""
        rng = random.Random(seed)
        pairs = list(combinations(branches, 2))
        # order pairs by scarcity (fewer available paths first)
        def pair_options(b):
            u, v = b
            opts = []
            for mask, p in paths[u][v]:
                opts.append((mask, p))
            for mask, p in paths[v][u]:
                opts.append((mask, list(reversed(p))))
            return opts
        pairs.sort(key=lambda b: len(pair_options(b)))
        used_mask = 0
        assign = {}
        # try a few random orderings among equal-scarcity
        for attempt in range(30):
            used_mask = 0
            assign = {}
            ok = True
            # shuffle pairs a bit but keep scarcity order roughly
            order = list(pairs)
            if attempt > 0:
                rng.shuffle(order)
            for b in order:
                u, v = b
                opts = pair_options(b)
                # prefer shortest / least-overlap
                rng.shuffle(opts)
                opts.sort(key=lambda t: (len(t[1]), bin(t[0] & used_mask).count('1')))
                placed = False
                for mask, p in opts:
                    if mask & used_mask == 0:
                        used_mask |= mask
                        assign[b] = p
                        placed = True
                        break
                if not placed:
                    ok = False
                    break
            if ok:
                return assign
        return None

    def valid_solution(branches, assign):
        if len(branches) < 2 or len(set(branches)) != len(branches):
            return False
        used = 0
        for b in combinations(branches, 2):
            p = assign.get(b)
            if p is None:
                return False
            if p[0] not in b or p[-1] not in b or p[0] == p[-1]:
                return False
            if len(set(p)) != len(p):
                return False
            for i in range(len(p) - 1):
                if p[i + 1] not in adj[p[i]]:
                    return False
            interior = p[1:-1]
            m = 0
            for x in interior:
                if x in branches:
                    return False
                m |= (1 << x)
            if m & used:
                return False
            used |= m
        return True

    # ---- search for largest feasible branch set ----
    # Candidate pool: all vertices. Use local search: start from clique-ish seeds, add/swap.

    def max_clique_bk():
        best = []

        def bk(R, P, X):
            nonlocal best
            if not check_time():
                return
            if not P and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            if P:
                pivot = max(P, key=lambda v: len(adj[v] & (P | X)))
            else:
                pivot = None
            cands = set(P)
            if pivot is not None:
                cands -= adj[pivot]
            for v in list(cands):
                if not check_time():
                    return
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)

        bk(set(), set(range(n)), set())
        return best

    best_branches = []
    best_assign = {}

    def consider(branches):
        nonlocal best_branches, best_assign
        if len(branches) <= len(best_branches):
            return
        for seed in range(40):
            if not check_time():
                return
            a = try_assign(branches, seed)
            if a is not None and valid_solution(branches, a):
                best_branches = list(branches)
                best_assign = a
                return

    # seed 1: max clique
    clique = max_clique_bk()
    if len(clique) >= 2:
        consider(clique)
    # seed 2: top-degree vertices
    by_deg = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    for k in (12, 10, 8, 6):
        if not check_time():
            break
        consider(by_deg[:k])

    # ---- local search: start from best, try add / 2-opt swap ----
    def local_search(branches, assign):
        nonlocal best_branches, best_assign
        cur = list(branches)
        cur_assign = dict(assign)
        improved = True
        while improved and check_time():
            improved = False
            # try additions
            cands = [i for i in range(n) if i not in cur]
            cands.sort(key=lambda x: len(adj[x]), reverse=True)
            for newb in cands:
                if not check_time():
                    break
                nb = cur + [newb]
                for seed in range(15):
                    a = try_assign(nb, seed)
                    if a is not None and valid_solution(nb, a):
                        cur = nb
                        cur_assign = a
                        improved = True
                        break
                if improved:
                    break
            # try 2-opt swaps (remove one, add one) if stuck
            if not improved and len(cur) >= 3:
                for ridx in range(len(cur)):
                    if not check_time():
                        break
                    rem = cur[ridx]
                    base = [x for i, x in enumerate(cur) if i != ridx]
                    cands = [i for i in range(n) if i not in base]
                    random.shuffle(cands)
                    for newb in cands:
                        if not check_time():
                            break
                        nb = base + [newb]
                        for seed in range(10):
                            a = try_assign(nb, seed)
                            if a is not None and valid_solution(nb, a):
                                if len(nb) > len(cur):
                                    cur = nb
                                    cur_assign = a
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
        if len(cur) > len(best_branches):
            best_branches = cur
            best_assign = cur_assign

    if len(best_branches) >= 2:
        local_search(best_branches, best_assign)

    # ---- random restarts to explore other basins ----
    for _ in range(40):
        if not check_time():
            break
        if not edges:
            break
        u, v = random.choice(edges)
        cur = [u, v]
        for seed in range(20):
            a = try_assign(cur, seed)
            if a is not None and valid_solution(cur, a):
                break
        # grow
        for _ in range(120):
            if not check_time():
                break
            cands = [i for i in range(n) if i not in cur]
            random.shuffle(cands)
            added = False
            for newb in cands[:25]:
                nb = cur + [newb]
                for seed in range(12):
                    a = try_assign(nb, seed)
                    if a is not None and valid_solution(nb, a):
                        cur = nb
                        added = True
                        break
                if added:
                    break
            if not added:
                break
        if len(cur) > len(best_branches):
            for seed in range(40):
                a = try_assign(cur, seed)
                if a is not None and valid_solution(cur, a):
                    best_branches = cur
                    best_assign = a
                    break
            local_search(best_branches, best_assign)

    # ---- fallback ----
    if len(best_branches) < 2:
        if edges:
            u, v = edges[0]
            best_branches = [u, v]
            best_assign = {(u, v): [u, v]}
        else:
            best_branches = [0, 1]
            best_assign = {(0, 1): [0, 1]}

    final_paths = []
    for u, v in combinations(best_branches, 2):
        p = best_assign.get((u, v))
        if p is None:
            p = best_assign.get((v, u))
            if p is not None:
                p = list(reversed(p))
        if p is None:
            # should not happen; build direct edge fallback
            if v in adj[u]:
                p = [u, v]
            else:
                p = [u, v]
        final_paths.append([int(x) for x in p])

    return {
        'branches': [int(x) for x in best_branches],
        'paths': final_paths,
    }


# EVOLVE-BLOCK-END
