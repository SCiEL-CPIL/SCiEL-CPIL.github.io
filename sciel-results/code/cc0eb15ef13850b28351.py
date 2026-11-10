"""Few colors without any bicolored cycle."""
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def construct():
    start = time.time()
    deadline = start + 75.0
    n = P['n']
    edges = P['edges']

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Order: highest degree first
    order = sorted(range(n), key=lambda v: (-len(adj[v]), v))
    # Map: position in order -> vertex
    # We assign in this order

    class DSU:
        __slots__ = ('parent', 'rank', 'hist')
        def __init__(self, size):
            self.parent = list(range(size))
            self.rank = [0] * size
            self.hist = []

        def find(self, x):
            r = x
            while self.parent[r] != r:
                r = self.parent[r]
            return r

        def union(self, x, y):
            rx = self.find(x)
            ry = self.find(y)
            if rx == ry:
                return False
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.hist.append((ry, rx, self.rank[rx]))
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True

        def undo(self):
            ry, rx, old_rank = self.hist.pop()
            self.parent[ry] = ry
            self.rank[rx] = old_rank

    def solve(k):
        color = [-1] * n
        # pair (a,b) a<b -> index
        pairs = {}
        idx = 0
        for a in range(k):
            for b in range(a + 1, k):
                pairs[(a, b)] = idx
                idx += 1
        dsus = [DSU(n) for _ in range(idx)]

        def pidx(a, b):
            if a > b:
                a, b = b, a
            return pairs[(a, b)]

        # Forward checking: for each unassigned vertex, track which colors are still possible
        # Simple version: just backtrack

        def bt(pos):
            if time.time() > deadline:
                return False
            if pos == n:
                return True
            v = order[pos]
            # Determine which colors are forbidden (neighbors already colored)
            forbidden = set()
            for u in adj[v]:
                if color[u] >= 0:
                    forbidden.add(color[u])
            # Try colors
            for c in range(k):
                if c in forbidden:
                    continue
                # Tentatively assign
                color[v] = c
                unions = []
                ok = True
                for u in adj[v]:
                    d = color[u]
                    if d >= 0 and d != c:
                        pi = pidx(c, d)
                        if not dsus[pi].union(u, v):
                            ok = False
                            break
                        unions.append(pi)
                if ok:
                    if bt(pos + 1):
                        return True
                for pi in reversed(unions):
                    dsus[pi].undo()
                color[v] = -1
            return False

        return bt(0)

    best = list(range(n))
    for k in range(2, n + 1):
        if time.time() > deadline:
            break
        color = [-1] * n
        # Re-run solve with fresh state
        # We need to restructure: solve modifies color
        # Let me restructure to return the coloring
        pass

    # Let me restructure properly
    def solve_k(k):
        color = [-1] * n
        pairs = {}
        idx = 0
        for a in range(k):
            for b in range(a + 1, k):
                pairs[(a, b)] = idx
                idx += 1
        dsus = [DSU(n) for _ in range(idx)]

        def pidx(a, b):
            if a > b:
                a, b = b, a
            return pairs[(a, b)]

        def bt(pos):
            if time.time() > deadline:
                return False
            if pos == n:
                return True
            v = order[pos]
            forbidden = set()
            for u in adj[v]:
                if color[u] >= 0:
                    forbidden.add(color[u])
            for c in range(k):
                if c in forbidden:
                    continue
                color[v] = c
                unions = []
                ok = True
                for u in adj[v]:
                    d = color[u]
                    if d >= 0 and d != c:
                        pi = pidx(c, d)
                        if not dsus[pi].union(u, v):
                            ok = False
                            break
                        unions.append(pi)
                if ok:
                    if bt(pos + 1):
                        return True
                for pi in reversed(unions):
                    dsus[pi].undo()
                color[v] = -1
            return False

        if bt(0):
            return color
        return None

    best = list(range(n))
    for k in range(2, n + 1):
        if time.time() > deadline:
            break
        res = solve_k(k)
        if res is not None:
            best = res
            break

    return [int(c) for c in best]
