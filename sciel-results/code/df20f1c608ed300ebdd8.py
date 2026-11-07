"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def construct():
    start = time.time()
    deadline = start + 84.0  # explicit safety margin before the 90s kill
    n = P['n']
    edges = P['edges']

    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    deg = [len(adj[i]) for i in range(n)]
    rng = random.Random(12345)

    # ---------------- fast validity helpers ----------------
    def has_bicolored_cycle(colors, c1, c2):
        verts = [i for i in range(n) if colors[i] in (c1, c2)]
        if len(verts) < 3:
            return False
        parent = {v: v for v in verts}
        vs = set(verts)
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for u in verts:
            for v in adj[u]:
                if v in vs and u < v:
                    ru, rv = find(u), find(v)
                    if ru == rv:
                        return True
                    parent[ru] = rv
        return False

    def is_valid(colors, k):
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        for i in range(k):
            for j in range(i + 1, k):
                if has_bicolored_cycle(colors, i, j):
                    return False
        return True

    def distinct_of(colors):
        return len(set(colors))

    # ---------------- fast local-search refinement ----------------
    def local_search(colors, k, budget):
        """Recolor vertices to reduce distinct colors while keeping validity."""
        cur = colors[:]
        curk = k
        best = cur[:]
        best_d = distinct_of(cur)
        no_improve = 0
        while time.time() < budget and no_improve < 600:
            v = rng.randint(0, n - 1)
            old = cur[v]
            best_nc = None
            best_nd = best_d
            for nc in range(curk):
                if nc == old:
                    continue
                # proper-coloring check
                ok = True
                for nb in adj[v]:
                    if cur[nb] == nc:
                        ok = False
                        break
                if not ok:
                    continue
                # bicolored-cycle check only for pairs involving nc
                bad = False
                for c2 in range(curk):
                    if c2 == nc:
                        continue
                    if has_bicolored_cycle(cur, nc, c2):
                        bad = True
                        break
                if bad:
                    continue
                # compute distinct count
                used = set(cur)
                used.discard(nc)
                if old not in used:
                    nd = len(used)
                else:
                    nd = len(used) + 1
                if nd < best_nd:
                    best_nd = nd
                    best_nc = nc
            if best_nc is not None:
                cur[v] = best_nc
                best_d = best_nd
                no_improve = 0
                if best_d < distinct_of(best):
                    best = cur[:]
            else:
                no_improve += 1
        return best

    # ---------------- backtracking with incremental DSU + forward checking ----------------
    def bt_solve(order, k, budget):
        """Return a valid coloring using <= k colors, or None."""
        m = len(order)
        colors = [-1] * n
        uf = {}
        for i in range(k):
            for j in range(i + 1, k):
                uf[(i, j)] = {}

        def find(pd, x):
            if x not in pd:
                pd[x] = x
                return x
            root = x
            while pd[root] != root:
                root = pd[root]
            # path halving
            while pd[x] != root:
                nxt = pd[x]
                pd[x] = root
                x = nxt
            return root

        def union(pd, a, b):
            ra, rb = find(pd, a), find(pd, b)
            if ra == rb:
                return False
            pd[ra] = rb
            return True

        nodes = [0]
        timed_out = [False]

        def bt(pos):
            if timed_out[0]:
                return False
            if nodes[0] % 1024 == 0 and time.time() > budget:
                timed_out[0] = True
                return False
            if pos == m:
                return True
            v = order[pos]
            # color domain
            if pos == 0:
                domain = [0]
            else:
                domain = list(range(k))
            # forward checking: find a future vertex with empty domain
            fc = None
            for q in range(pos + 1, m):
                w = order[q]
                dom = 0
                for c in range(k):
                    conflict = False
                    for nb in adj[w]:
                        if colors[nb] == c:
                            conflict = True
                            break
                    if not conflict:
                        dom += 1
                if dom == 0:
                    fc = True
                    break
            if fc:
                return False
            for c in domain:
                # proper coloring
                ok = True
                for nb in adj[v]:
                    if colors[nb] == c:
                        ok = False
                        break
                if not ok:
                    continue
                # bicolored cycle pre-check
                conflict = False
                for c2 in range(k):
                    if c2 == c:
                        continue
                    roots = set()
                    for nb in adj[v]:
                        if colors[nb] == c2:
                            r = find(uf[(min(c, c2), max(c, c2))], nb)
                            if r in roots:
                                conflict = True
                                break
                            roots.add(r)
                    if conflict:
                        break
                if conflict:
                    continue
                # assign
                colors[v] = c
                undo = []
                all_ok = True
                for c2 in range(k):
                    if c2 == c:
                        continue
                    pk = (min(c, c2), max(c, c2))
                    pd = uf[pk]
                    for nb in adj[v]:
                        if colors[nb] == c2:
                            rv = find(pd, v)
                            rn = find(pd, nb)
                            if rv == rn:
                                all_ok = False
                                break
                            undo.append((pk, rv, pd.get(rv)))
                            pd[rv] = rn
                    if not all_ok:
                        break
                if all_ok:
                    nodes[0] += 1
                    if bt(pos + 1):
                        return True
                # undo
                for pk, rv, ov in reversed(undo):
                    pd = uf[pk]
                    if ov is None:
                        if rv in pd:
                            del pd[rv]
                    else:
                        pd[rv] = ov
                colors[v] = -1
            return False

        if bt(0):
            return colors
        return None

    # ---------------- main search ----------------
    best_colors = list(range(n))
    best_score = n

    # vertex orderings (most-constrained first prunes best)
    orderings = [
        sorted(range(n), key=lambda x: (-deg[x], x)),
        list(range(n)),
        list(range(n))[::-1],
        sorted(range(n), key=lambda x: x % 7),
    ]

    # Try increasing k; for each, several orderings until solved or time up.
    for k in range(3, 11):
        if time.time() > deadline - 12:
            break
        solved = False
        for oi, order in enumerate(orderings):
            if time.time() > deadline - 8:
                break
            budget = deadline - 6 - oi * 1.0
            res = bt_solve(order, k, budget)
            if res is not None and is_valid(res, k):
                d = distinct_of(res)
                if d < best_score:
                    best_score = d
                    best_colors = res[:]
                solved = True
                break
        if solved:
            # refine with local search to try to squeeze more colors out
            if time.time() < deadline - 4:
                refined = local_search(best_colors, best_score, deadline - 2)
                rk = distinct_of(refined)
                if is_valid(refined, rk) and rk < best_score:
                    best_score = rk
                    best_colors = refined[:]
            break

    # fallback: ensure we always return something valid
    if not is_valid(best_colors, distinct_of(best_colors)):
        best_colors = list(range(n))
        best_score = n

    return [int(c) for c in best_colors]
# EVOLVE-BLOCK-END
