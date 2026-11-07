import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

class ReversibleUnionFind:
    __slots__ = ('parent', 'rank', 'undo_stack')
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.undo_stack = []

    def find(self, x):
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, x, y):
        p = self.parent
        r = self.rank
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        self.undo_stack.append((rx, ry, p[ry], r[rx], r[ry]))
        if r[rx] < r[ry]:
            rx, ry = ry, rx
        p[ry] = rx
        if r[rx] == r[ry]:
            r[rx] += 1
        return True

    def undo(self):
        if not self.undo_stack:
            return
        rx, ry, old_parent_ry, old_rank_rx, old_rank_ry = self.undo_stack.pop()
        self.parent[ry] = old_parent_ry
        self.rank[rx] = old_rank_rx
        self.rank[ry] = old_rank_ry

    def snapshot(self):
        return len(self.undo_stack)

    def rollback(self, snap):
        while len(self.undo_stack) > snap:
            self.undo()


def _is_valid(coloring, edges, n):
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu != cv:
            pair_edges[(min(cu, cv), max(cu, cv))].append((u, v))
    for elist in pair_edges.values():
        uf = ReversibleUnionFind(n)
        for u, v in elist:
            if not uf.union(u, v):
                return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 80
    n = P['n']
    edges = P['edges']

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Chord = edge not on the main 50-cycle (i, i+1 mod 50). Chord-rich vertices
    # are the most constrained, so they drive ordering and symmetry breaking.
    chord_count = [0] * n
    for u, v in edges:
        d = (v - u) % n
        if d not in (1, n - 1):
            chord_count[u] += 1
            chord_count[v] += 1
    order_index = {v: i for i, v in enumerate(
        sorted(range(n), key=lambda v: (-chord_count[v], v)))}

    def find_coloring(k):
        coloring = [-1] * n
        assigned = [False] * n
        pair_uf = {}

        def get_uf(c1, c2):
            key = (min(c1, c2), max(c1, c2))
            uf = pair_uf.get(key)
            if uf is None:
                uf = ReversibleUnionFind(n)
                pair_uf[key] = uf
            return uf

        # Assign v->c. Returns (ok, snapshots). snapshots: list of (uf, snap_before).
        def try_assign(v, c):
            snapshots = []
            for nb in adj[v]:
                cn = coloring[nb]
                if cn == -1:
                    continue
                uf = get_uf(c, cn)
                snap = uf.snapshot()
                if not uf.union(v, nb):
                    # cycle would form: roll back everything we did in this call
                    for ufk, s in snapshots:
                        ufk.rollback(s)
                    return (False, None)
                snapshots.append((uf, snap))
            return (True, snapshots)

        def unassign(v, snapshots):
            for uf, s in snapshots:
                uf.rollback(s)

        def backtrack():
            if time.time() > deadline:
                return False
            # MRV: pick unassigned vertex with fewest available colors.
            best_v = -1
            best_avail = None
            best_count = k + 1
            for v in range(n):
                if assigned[v]:
                    continue
                used = set()
                for nb in adj[v]:
                    cn = coloring[nb]
                    if cn != -1:
                        used.add(cn)
                avail = [c for c in range(k) if c not in used]
                cnt = len(avail)
                if cnt == 0:
                    return False  # dead end
                if cnt < best_count:
                    best_count = cnt
                    best_v = v
                    best_avail = avail
                    if cnt == 1:
                        break
            if best_v == -1:
                return True  # all assigned

            v = best_v
            # least-constraining: prefer colors used less among v's neighbors' neighbors
            def score(c):
                s = 0
                for nb in adj[v]:
                    for nb2 in adj[nb]:
                        if coloring[nb2] == c:
                            s += 1
                return s
            avail = sorted(best_avail, key=score)

            for c in avail:
                ok, snapshots = try_assign(v, c)
                if not ok:
                    continue
                coloring[v] = c
                assigned[v] = True
                if backtrack():
                    return True
                unassign(v, snapshots)
                assigned[v] = False
                coloring[v] = -1
            return False

        # Symmetry breaking: pin the first (most chord-rich) vertex to color 0.
        v0 = order_index and min(range(n), key=lambda v: order_index[v])
        v0 = sorted(range(n), key=lambda v: (-chord_count[v], v))[0]
        coloring[v0] = 0
        assigned[v0] = True
        res = backtrack()
        if res:
            return coloring
        return None

    for k in range(3, 13):
        if time.time() > deadline - 5:
            break
        coloring = find_coloring(k)
        if coloring is not None and _is_valid(coloring, edges, n):
            return [int(c) for c in coloring]

    # Fallback: all-distinct coloring is always valid (each pair subgraph is a forest).
    return list(range(n))
