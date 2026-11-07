"""Bicolored-forest coloring via direct structural analysis."""
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

adj = [[] for _ in range(N)]
for u, v in EDGES:
    adj[u].append(v)
    adj[v].append(u)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


def valid_coloring(colors):
    """Check adjacent-differ and that every 2-color induced subgraph is a forest."""
    for u, v in EDGES:
        if colors[u] == colors[v]:
            return False
    k = max(colors) + 1
    color_to_verts = [[] for _ in range(k)]
    for i, c in enumerate(colors):
        color_to_verts[c].append(i)
    for a in range(k):
        for b in range(a + 1, k):
            verts = set(color_to_verts[a]) | set(color_to_verts[b])
            if len(verts) <= 1:
                continue
            vert_list = list(verts)
            vert_to_idx = {v: i for i, v in enumerate(vert_list)}
            uf = UnionFind(len(vert_list))
            for u in vert_list:
                for w in adj[u]:
                    if w in vert_to_idx and u < w:
                        if not uf.union(vert_to_idx[u], vert_to_idx[w]):
                            return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 80

    # Candidate colorings, ordered by number of distinct colors (fewest first).
    candidates = []
    for period in range(2, 13):
        for offset in range(period):
            c = [(i + offset) % period for i in range(N)]
            candidates.append(c)
    # Also include a couple of structured non-periodic candidates.
    candidates.append(list(range(N)))

    # Deduplicate by canonical label set (relabeled to 0..k-1 in order of first use)
    seen = set()
    unique_candidates = []
    for c in candidates:
        mapping = {}
        canon = []
        next_id = 0
        for x in c:
            if x not in mapping:
                mapping[x] = next_id
                next_id += 1
            canon.append(mapping[x])
        key = tuple(canon)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(canon)

    best = None
    best_count = N + 1

    for c in unique_candidates:
        if time.time() > deadline:
            break
        if not valid_coloring(c):
            continue
        cnt = max(c) + 1
        if cnt < best_count:
            best_count = cnt
            best = list(c)
            if best_count <= 2:
                break

    if best is None:
        # Fallback: 50 distinct colors is always valid (no bicolored subgraph has >1 color)
        return [int(i) for i in range(N)]

    return [int(x) for x in best]
