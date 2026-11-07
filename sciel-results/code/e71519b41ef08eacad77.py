"""Rewritten program with a stronger structural approach:
- Exact maximum clique via branch-and-bound
- Greedy + local-search branch-set extension
- Efficient CSP solver with MRV and forward checking for disjoint paths
- Precomputed short paths for speed
"""
import time
import random
import sys
from collections import deque

sys.setrecursionlimit(100000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        adj_list[u].append(v)
        adj_list[v].append(u)

    degrees = [len(adj[i]) for i in range(n)]
    start_time = time.time()
    deadline = start_time + 85.0

    # Precompute all-pairs BFS distances and predecessors
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    pred = [[-1] * n for _ in range(n)]
    for s in range(n):
        dist[s][s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj_list[u]:
                if dist[s][v] == INF:
                    dist[s][v] = dist[s][u] + 1
                    pred[s][v] = u
                    q.append(v)

    def get_shortest_path(s, t):
        if dist[s][t] == INF:
            return None
        path = []
        cur = t
        while cur != -1:
            path.append(cur)
            cur = pred[s][cur]
        path.reverse()
        return path

    # Precompute simple paths of length 2..5 for every pair (cached)
    path_cache = {}

    def get_paths(u, v, max_len=5):
        key = (min(u, v), max(u, v), max_len)
        if key in path_cache:
            return path_cache[key]
        if u == v:
            path_cache[key] = [[u]]
            return path_cache[key]
        paths = []

        def dfs(curr, path, visited):
            if len(path) - 1 > max_len:
                return
            if curr == v:
                paths.append(list(path))
                return
            for nxt in adj_list[curr]:
                if nxt not in visited:
                    visited.add(nxt)
                    path.append(nxt)
                    dfs(nxt, path, visited)
                    path.pop()
                    visited.remove(nxt)

        dfs(u, [u], {u})
        paths.sort(key=lambda p: (len(p), p))
        path_cache[key] = paths
        return paths

    # Maximum clique via branch-and-bound (Bron-Kerbosch with pivot)
    best_clique = []

    def bron_kerbosch(R, Pset, X):
        nonlocal best_clique
        if not Pset and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        if Pset:
            pivot = max(Pset, key=lambda v: len(adj[v] & Pset))
            candidates = Pset - adj[pivot]
        else:
            candidates = Pset
        for v in list(candidates):
            bron_kerbosch(R | {v}, Pset & adj[v], X & adj[v])
            Pset.remove(v)
            X.add(v)

    bron_kerbosch(set(), set(range(n)), set())
    max_clique_size = len(best_clique)

    # CSP solver: given branch set B, find disjoint paths for all non-adjacent pairs
    def solve_paths(B, time_limit=3.0):
        start_t = time.time()
        B_set = set(B)
        k = len(B)

        # Identify non-adjacent pairs needing paths
        non_adj = []
        for i in range(k):
            for j in range(i + 1, k):
                u, v = B[i], B[j]
                if v not in adj[u]:
                    non_adj.append((u, v))

        if not non_adj:
            return True, {}

        # Build domains: for each pair, candidate paths avoiding B in interior
        domains = {}
        for u, v in non_adj:
            cands = []
            for L in range(2, 7):
                ps = get_paths(u, v, L)
                valid = [p for p in ps if not any(x in B_set for x in p[1:-1])]
                if valid:
                    cands.extend(valid)
                    break
            if not cands:
                ps = get_paths(u, v, 8)
                valid = [p for p in ps if not any(x in B_set for x in p[1:-1])]
                if not valid:
                    return False, None
                cands = valid
            domains[(u, v)] = cands

        # Order by MRV (smallest domain first)
        ordered = sorted(non_adj, key=lambda p: len(domains[p]))

        used = set()
        assignment = {}

        def backtrack(idx):
            if time.time() - start_t > time_limit:
                return False
            if idx == len(ordered):
                return True
            u, v = ordered[idx]
            for path in domains[(u, v)]:
                interior = path[1:-1]
                if any(w in used for w in interior):
                    continue
                used.update(interior)
                assignment[(u, v)] = path
                if backtrack(idx + 1):
                    return True
                used.difference_update(interior)
                del assignment[(u, v)]
            return False

        if backtrack(0):
            return True, dict(assignment)
        return False, None

    # Greedy build from max clique
    best_B = list(best_clique)
    best_paths = {}
    ok, best_paths = solve_paths(best_B, time_limit=5.0)
    if not ok:
        # Shrink until valid
        while len(best_B) > 2:
            best_B.pop()
            ok, best_paths = solve_paths(best_B, time_limit=3.0)
            if ok:
                break
        if not ok:
            best_B = [best_B[0], best_B[1]]
            best_paths = {}

    # Greedy extension
    while time.time() < deadline - 15:
        B_set = set(best_B)
        candidates = [v for v in range(n) if v not in B_set]
        candidates.sort(key=lambda v: -len(adj[v] & B_set))
        extended = False
        for v in candidates:
            new_B = best_B + [v]
            ok, new_paths = solve_paths(new_B, time_limit=2.0)
            if ok:
                best_B = new_B
                best_paths = new_paths
                extended = True
                break
        if not extended:
            break

    # Local search: try swaps to unlock further growth
    if time.time() < deadline - 10:
        for _ in range(30):
            if time.time() > deadline - 10:
                break
            B_set = set(best_B)
            # Try removing one vertex and adding a different one
            for i in range(len(best_B)):
                if time.time() > deadline - 10:
                    break
                candidates = [v for v in range(n) if v not in B_set]
                candidates.sort(key=lambda v: -degrees[v])
                for v in candidates[:8]:
                    new_B = best_B[:i] + [v] + best_B[i + 1:]
                    ok, new_paths = solve_paths(new_B, time_limit=1.0)
                    if ok:
                        best_B = new_B
                        best_paths = new_paths
                        break
                else:
                    continue
                break
            # After swap, try to extend again
            B_set = set(best_B)
            candidates = [v for v in range(n) if v not in B_set]
            candidates.sort(key=lambda v: -len(adj[v] & B_set))
            for v in candidates[:5]:
                new_B = best_B + [v]
                ok, new_paths = solve_paths(new_B, time_limit=1.5)
                if ok:
                    best_B = new_B
                    best_paths = new_paths
                    break

    # Random restarts for larger k
    if time.time() < deadline - 5:
        for target_k in range(len(best_B) + 1, n + 1):
            if time.time() > deadline - 5:
                break
            found = False
            for attempt in range(40):
                if time.time() > deadline - 5:
                    break
                B_rand = random.sample(range(n), target_k)
                ok, paths_rand = solve_paths(B_rand, time_limit=1.5)
                if ok:
                    best_B = B_rand
                    best_paths = paths_rand
                    found = True
                    break
            if not found:
                break

    # Final verification
    B = list(best_B)
    if len(B) < 2:
        B = [0, 1]

    ok, paths_dict = solve_paths(B, time_limit=3.0)
    if not ok:
        # Shrink
        while len(B) > 2:
            B.pop()
            ok, paths_dict = solve_paths(B, time_limit=2.0)
            if ok:
                break
        if not ok:
            B = [0, 1]
            paths_dict = {}

    # Build output
    B_set = set(B)
    paths_out = []
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            u, v = B[i], B[j]
            if v in adj[u]:
                paths_out.append([u, v])
            else:
                if (u, v) in paths_dict:
                    paths_out.append(list(paths_dict[(u, v)]))
                elif (v, u) in paths_dict:
                    paths_out.append(list(reversed(paths_dict[(v, u)])))
                else:
                    p = get_shortest_path(u, v)
                    if p:
                        paths_out.append(p)
                    else:
                        paths_out.append([u, v])

    branches_out = [int(b) for b in B]
    paths_out = [[int(x) for x in p] for p in paths_out]

    return {'branches': branches_out, 'paths': paths_out}
