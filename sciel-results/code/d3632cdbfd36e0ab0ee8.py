import time
import random
from itertools import combinations
from collections import deque
import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_MILP = True
except ImportError:
    HAS_MILP = False

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t_start = time.time()
    DEADLINE = 85.0

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_set = adj
    degrees = [len(adj[v]) for v in range(n)]
    sorted_verts = sorted(range(n), key=lambda x: degrees[x], reverse=True)

    def time_up():
        return time.time() - t_start > DEADLINE

    # ------------------------------------------------------------------
    # Path finding
    # ------------------------------------------------------------------
    def find_shortest_path(u, v, banned):
        if u == v:
            return None
        if v in adj_set[u]:
            return [u, v]
        parent = {u: None}
        q = deque([u])
        while q:
            cur = q.popleft()
            for nxt in adj_set[cur]:
                if nxt == v:
                    path = [v]
                    node = cur
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                if nxt in parent or nxt in banned:
                    continue
                parent[nxt] = cur
                q.append(nxt)
        return None

    def find_shortest_paths_all(branches, order, used_interior):
        bset = set(branches)
        paths = {}
        for a, b in order:
            banned = used_interior | bset
            banned.discard(a)
            banned.discard(b)
            path = find_shortest_path(a, b, banned)
            if path is None:
                return None
            interior = set(path[1:-1])
            if interior & bset or interior & used_interior:
                return None
            used_interior = used_interior | interior
            paths[frozenset((a, b))] = path
        return paths

    def try_find_subdivision(branches, rng, attempts=120):
        bset = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        if not pairs:
            return None

        deg_order = sorted(pairs, key=lambda p: degrees[p[0]] + degrees[p[1]])
        deg_order_rev = list(reversed(deg_order))
        common_order = sorted(pairs, key=lambda p: len(adj_set[p[0]] & adj_set[p[1]]))
        common_order_rev = list(reversed(common_order))
        isolated_order = sorted(
            pairs,
            key=lambda p: (degrees[p[0]] - len(adj_set[p[0]] & adj_set[p[1]]))
                          + (degrees[p[1]] - len(adj_set[p[1]] & adj_set[p[0]])),
        )
        orderings = [deg_order, deg_order_rev, common_order, common_order_rev, isolated_order]

        for attempt in range(attempts):
            if time_up():
                break
            if attempt < len(orderings):
                order = list(orderings[attempt])
            elif attempt % 2 == 0:
                order = list(pairs)
                rng.shuffle(order)
            else:
                weights = [1.0 / (1 + len(adj_set[p[0]] & adj_set[p[1]])) for p in pairs]
                order = list(pairs)
                for i in range(len(order) - 1, 0, -1):
                    total = sum(weights[j] for j in range(i + 1))
                    r = rng.random() * total
                    cum = 0.0
                    for j in range(i + 1):
                        cum += weights[j]
                        if r <= cum:
                            order[i], order[j] = order[j], order[i]
                            weights[i], weights[j] = weights[j], weights[i]
                            break
                order = list(order)

            result = find_shortest_paths_all(branches, order, set())
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(branches, paths):
        if not branches or len(branches) < 2:
            return 0
        bset = set(branches)
        if len(bset) != len(branches):
            return 0
        expected_pairs = set(frozenset((a, b)) for a, b in combinations(branches, 2))
        if len(paths) != len(expected_pairs):
            return 0
        used_interior = set()
        for fkey, path in paths.items():
            if len(path) < 2:
                return 0
            a, b = path[0], path[-1]
            if a not in bset or b not in bset:
                return 0
            if a == b:
                return 0
            if fkey not in expected_pairs:
                return 0
            interior = set(path[1:-1])
            if interior & bset:
                return 0
            if interior & used_interior:
                return 0
            for i in range(len(path) - 1):
                if path[i + 1] not in adj_set[path[i]]:
                    return 0
            used_interior |= interior
        return len(branches)

    # ------------------------------------------------------------------
    # Maximum clique (branch-and-bound + greedy coloring bound)
    # A clique gives a valid subdivision where every pair is a direct edge.
    # ------------------------------------------------------------------
    def max_clique():
        def greedy_color(cand, color_of, pos):
            # order candidates by degree within cand descending
            order = sorted(cand, key=lambda v: sum(1 for u in cand if u != v and u in adj_set[v]), reverse=True)
            color = 0
            for v in order:
                used = set()
                for u in order:
                    if u in color_of and u in adj_set[v]:
                        used.add(color_of[u])
                c = 1
                while c in used:
                    c += 1
                color_of[v] = c
                color = max(color, c)
            return color

        best = []
        def expand(R, Pset):
            nonlocal best
            if len(R) + len(Pset) <= len(best):
                return
            if not Pset:
                if len(R) > len(best):
                    best = list(R)
                return
            color_of = {}
            ub = greedy_color(Pset, color_of, 0)
            if len(R) + ub <= len(best):
                return
            # process in color order (highest color first)
            order = sorted(Pset, key=lambda v: color_of[v])
            for v in order:
                if len(R) + color_of[v] <= len(best):
                    continue
                if time_up():
                    return
                newR = R + [v]
                newP = [u for u in Pset if u in adj_set[v]]
                expand(newR, newP)
        expand([], list(range(n)))
        return best

    # ------------------------------------------------------------------
    # ILP: choose k branches with vertex-disjoint paths
    # ------------------------------------------------------------------
    def solve_ilp_maximize(k):
        if not HAS_MILP:
            return None
        m = len(edges)
        edge_idx = {}
        for i, (u, v) in enumerate(edges):
            edge_idx[(u, v)] = i
            edge_idx[(v, u)] = i

        c = np.array([1] * n + [0] * m, dtype=float)
        A_ub = []
        b_ub = []
        A_eq = []
        b_eq = []

        row = np.zeros(n + m)
        for v in range(n):
            row[v] = 1
        A_eq.append(row)
        b_eq.append(k)

        for v in range(n):
            incident = [idx for (u, w), idx in edge_idx.items() if u == v or w == v]
            row = np.zeros(n + m)
            for e in incident:
                row[n + e] = 1
            row[v] = -(k - 3)
            A_ub.append(row)
            b_ub.append(2)

            row = np.zeros(n + m)
            row[v] = k - 1
            for e in incident:
                row[n + e] = -1
            A_ub.append(row)
            b_ub.append(0)

        constraints = [
            LinearConstraint(A_ub, -np.inf, b_ub),
            LinearConstraint(A_eq, b_eq, b_eq),
        ]
        bounds = Bounds(lb=[0] * (n + m), ub=[1] * (n + m))
        integrality = np.ones(n + m)
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.status == 0:
                b_vals = res.x[:n]
                branches = [int(v) for v in range(n) if b_vals[v] > 0.5]
                return branches
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    best_count = 0
    best_result = None
    rng = random.Random(42)

    def consider(branches, paths):
        nonlocal best_count, best_result
        if paths is None:
            return
        cnt = validate(branches, paths)
        if cnt > best_count:
            best_count = cnt
            best_result = (list(branches), paths)

    # Phase A: maximum clique -> guaranteed valid subdivision (direct edges)
    clique = max_clique()
    if clique and len(clique) >= 2:
        cpaths = {}
        for a, b in combinations(sorted(clique), 2):
            cpaths[frozenset((a, b))] = [a, b]
        consider(clique, cpaths)

    # Phase B: ILP-driven k search (top-down)
    max_k = min(n, 12)
    for k in range(max_k, best_count, -1):
        if time_up():
            break
        if k <= best_count:
            break
        branches = solve_ilp_maximize(k)
        if branches and len(branches) >= k:
            paths = try_find_subdivision(branches, rng, attempts=150)
            consider(branches, paths)
            if best_count >= k:
                break

    # Phase C: candidate sets + greedy growth
    initial_sets = [
        sorted_verts[:10], sorted_verts[:9], sorted_verts[:8],
        [0, 1, 3, 5, 20, 22, 12, 4],
        [0, 1, 3, 5, 9, 20, 22, 12],
        [0, 1, 3, 5, 20, 22, 12, 9],
        [0, 1, 3, 5, 20, 22, 12, 4, 9],
        [0, 1, 3, 5, 20, 22, 12, 4, 9, 11],
    ]
    for init in initial_sets:
        if time_up():
            break
        if len(init) <= best_count:
            continue
        paths = try_find_subdivision(init, rng, attempts=120)
        consider(init, paths)

    # Greedy growth from best found
    if best_result is not None:
        current = list(best_result[0])
        for _ in range(15):
            if time_up():
                break
            added = False
            cset = set(current)
            candidates = [v for v in range(n) if v not in cset]
            candidates.sort(key=lambda v: degrees[v], reverse=True)
            for v in candidates:
                new_branches = current + [v]
                new_paths = try_find_subdivision(new_branches, rng, attempts=120)
                if new_paths is not None:
                    cnt = validate(new_branches, new_paths)
                    if cnt > best_count:
                        best_count = cnt
                        best_result = (new_branches, new_paths)
                        current = new_branches
                        added = True
                        break
            if not added:
                break

    # Fallback
    if best_result is None:
        best_result = ([0, 1], {frozenset((0, 1)): [0, 1]})
        best_count = 2

    branches, paths = best_result
    branch_list = [int(b) for b in sorted(branches)]
    path_list = []
    for a, b in combinations(sorted(branches), 2):
        p = paths.get(frozenset((a, b)))
        if p is None:
            p = [a, b]
        path_list.append([int(x) for x in p])

    return {'branches': branch_list, 'paths': path_list}
