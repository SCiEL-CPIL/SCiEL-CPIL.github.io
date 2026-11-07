"""Combined solver: modular-pattern pre-scan + incremental DSU backtracking."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']
ADJ = [[] for _ in range(N)]
for u, v in EDGES:
    ADJ[u].append(v)
    ADJ[v].append(u)


def _has_bicolored_cycle(coloring, edges):
    """Check if any pair of colors induces a cycle using union-find."""
    n = len(coloring)
    color_map = {}
    for i, c in enumerate(coloring):
        if c not in color_map:
            color_map[c] = []
        color_map[c].append(i)
    colors = list(color_map.keys())
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            ci, cj = colors[i], colors[j]
            parent = {}

            def find(x):
                if x not in parent:
                    parent[x] = x
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return True
                parent[rx] = ry
                return False

            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == ci and cv == cj) or (cu == cj and cv == ci):
                    if union(u, v):
                        return True
    return False


def _is_valid(coloring, edges):
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    return not _has_bicolored_cycle(coloring, edges)


def _count_colors(coloring):
    return len(set(coloring))


def _try_backtrack(k, deadline):
    """Backtracking with incremental per-pair DSU (with full save/restore) and
    forward checking. Returns a valid k-coloring list, or None."""
    color = [-1] * N
    pair_parent = {}
    pair_rank = {}

    def pk(a, b):
        if a > b:
            a, b = b, a
        return (a, b)

    def find(p, v):
        par = pair_parent[p]
        while par[v] != v:
            par[v] = par[par[v]]
            v = par[v]
        return v

    def union(p, u, v):
        par = pair_parent[p]
        rnk = pair_rank[p]
        ru = find(p, u)
        rv = find(p, v)
        if ru == rv:
            return None
        if rnk[ru] < rnk[rv]:
            ru, rv = rv, ru
        par[rv] = ru
        if rnk[ru] == rnk[rv]:
            rnk[ru] += 1
        return (rv, ru)

    def init_pair(p, a, b):
        par = {}
        rnk = {}
        for w in range(N):
            cw = color[w]
            if cw == a or cw == b:
                par[w] = w
                rnk[w] = 0
        pair_parent[p] = par
        pair_rank[p] = rnk

    def assign(v, c):
        color[v] = c
        for u in ADJ[v]:
            d = color[u]
            if d == -1 or d == c:
                continue
            p = pk(c, d)
            if p not in pair_parent:
                init_pair(p, c, d)
            par = pair_parent[p]
            if v not in par:
                par[v] = v
                pair_rank[p][v] = 0
            res = union(p, v, u)
            if res is None:
                return None
        return [(c, u) for u in ADJ[v] if color[u] != -1 and color[u] != c]

    def unassign(v, c, undo):
        color[v] = -1
        for (d, u) in undo:
            p = pk(c, d)
            par = pair_parent[p]
            rnk = pair_rank[p]
            rv, ru = undo_pair_entry[p] if False else (None, None)
            # restore the union that was performed
            # we stored the (child_root, parent_root) in the return; re-derive by
            # using the saved value
            pass
        # (handled below via saved entries)
        if v in pair_parent:
            pass

    # Re-implement unassign using saved entries stored on the undo list
    undo_pair_entry = {}

    def assign2(v, c):
        color[v] = c
        saved = []
        for u in ADJ[v]:
            d = color[u]
            if d == -1 or d == c:
                continue
            p = pk(c, d)
            if p not in pair_parent:
                init_pair(p, c, d)
            par = pair_parent[p]
            rnk = pair_rank[p]
            if v not in par:
                par[v] = v
                rnk[v] = 0
                saved.append((p, v, 'init'))
            res = union(p, v, u)
            if res is None:
                # roll back the ones we did in this assign
                for entry in reversed(saved):
                    _roll(entry)
                color[v] = -1
                return None
            saved.append((p, res[0], res[1]))
        return saved

    def _roll(entry):
        if entry[2] == 'init':
            p, v, _ = entry
            pair_parent[p].pop(v, None)
            pair_rank[p].pop(v, None)
        else:
            p, rv, ru = entry
            par = pair_parent[p]
            rnk = pair_rank[p]
            par[rv] = rv
            if rnk[ru] > 0:
                rnk[ru] -= 1

    def unassign2(v, c, saved):
        for entry in reversed(saved):
            _roll(entry)
        color[v] = -1

    def backtrack(v):
        if time.time() > deadline:
            return False
        if v == N:
            return True
        # forward checking: colors still available at v
        avail = [c for c in range(k)
                 if all(color[u] != c for u in ADJ[v])]
        if not avail:
            return False
        for c in avail:
            saved = assign2(v, c)
            if saved is None:
                continue
            if backtrack(v + 1):
                return True
            unassign2(v, c, saved)
        return False

    if backtrack(0):
        return color[:]
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before the 90s kill
    n = N
    edges = EDGES

    best = list(range(n))
    best_count = n

    # 1) Fast modular-pattern pre-scan (cheap, may find a good solution instantly)
    for k in range(3, 12):
        if time.time() > deadline:
            break
        for a in range(1, k):
            if time.time() > deadline:
                break
            for b in range(k):
                if time.time() > deadline:
                    break
                coloring = [(i * a + b) % k for i in range(n)]
                if _is_valid(coloring, edges):
                    cnt = _count_colors(coloring)
                    if cnt < best_count:
                        best = coloring
                        best_count = cnt
                    if best_count == 3:
                        return [int(x) for x in best]

    # 2) Structural: incremental DSU backtracking to search for small-k colorings
    for k in range(3, best_count + 1):
        if time.time() > deadline:
            break
        res = _try_backtrack(k, deadline)
        if res is not None:
            cnt = _count_colors(res)
            if cnt < best_count:
                best = res
                best_count = cnt
            if best_count == 3:
                break

    return [int(x) for x in best]
# EVOLVE-BLOCK-END
