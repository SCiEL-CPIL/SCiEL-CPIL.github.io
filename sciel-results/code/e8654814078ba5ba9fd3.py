import time
import random

def construct():
    start = time.time()
    deadline = start + 110  # safety margin

    def popcount(x):
        return bin(x).count('1')

    def groetzsch():
        cycle = [(i, (i + 1) % 5) for i in range(5)]
        edges = [list(e) for e in cycle]
        for u, w in cycle:
            edges += [[u, 5 + w], [w, 5 + u]]
        return edges + [[5 + u, 10] for u in range(5)]

    def build_clebsch():
        edges = []
        for i in range(16):
            for j in range(i + 1, 16):
                if popcount(i ^ j) == 1 or popcount((31 - i) ^ j) == 1:
                    edges.append([i, j])
        return edges

    def make_adj(edges):
        adj = [[False] * 16 for _ in range(16)]
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
        return adj

    def is_triangle_free(adj):
        for u in range(16):
            for v in range(u + 1, 16):
                if adj[u][v]:
                    for w in range(v + 1, 16):
                        if adj[u][w] and adj[v][w]:
                            return False
        return True

    def is_3_colorable(adj):
        color = [-1] * 16
        neigh = [[u for u in range(16) if adj[v][u]] for v in range(16)]
        order = sorted(range(16), key=lambda v: -len(neigh[v]))
        pos = [0] * 16
        for i, v in enumerate(order):
            pos[v] = i
        # only need to check neighbors that come earlier in order
        earlier_neigh = [[] for _ in range(16)]
        for v in range(16):
            for u in neigh[v]:
                if pos[u] < pos[v]:
                    earlier_neigh[v].append(u)
        
        def backtrack(idx):
            if idx == 16:
                return True
            v = order[idx]
            used = [False, False, False]
            for u in earlier_neigh[v]:
                used[color[u]] = True
            for c in range(3):
                if not used[c]:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        return backtrack(0)

    # Build and verify Clebsch
    edges = build_clebsch()
    adj = make_adj(edges)

    if not (is_triangle_free(adj) and not is_3_colorable(adj)):
        g = groetzsch()
        ga = make_adj(g)
        if is_triangle_free(ga) and not is_3_colorable(ga):
            return g
        return edges

    best_edges = [list(e) for e in edges]
    best_count = len(edges)

    rng = random.Random(42)
    cur_adj = [row[:] for row in adj]
    cur_count = best_count

    def edges_of(adj):
        return [(u, v) for u in range(16) for v in range(u + 1, 16) if adj[u][v]]

    def nonedges_of(adj):
        return [(u, v) for u in range(16) for v in range(u + 1, 16) if not adj[u][v]]

    def valid(adj):
        return is_triangle_free(adj) and not is_3_colorable(adj)

    def step():
        nonlocal cur_adj, cur_count, best_edges, best_count
        el = edges_of(cur_adj)
        nl = nonedges_of(cur_adj)
        if not el or not nl:
            return
        rem = rng.choice(el)
        # choose number of additions: 1 (swap) or 2 (expand)
        if rng.random() < 0.5:
            k = 1
        else:
            k = 2
        adds = rng.sample(nl, min(k, len(nl)))
        new_adj = [row[:] for row in cur_adj]
        new_adj[rem[0]][rem[1]] = new_adj[rem[1]][rem[0]] = False
        for (u, v) in adds:
            new_adj[u][v] = new_adj[v][u] = True
        if not valid(new_adj):
            return
        new_count = cur_count - 1 + len(adds)
        # accept if improves, or (SA) if same/better with some probability
        if new_count >= cur_count:
            cur_adj = new_adj
            cur_count = new_count
            if new_count > best_count:
                best_count = new_count
                best_edges = [[u, v] for (u, v) in edges_of(new_adj)]

    iters = 0
    while time.time() < deadline and iters < 200000:
        step()
        iters += 1

    return best_edges
