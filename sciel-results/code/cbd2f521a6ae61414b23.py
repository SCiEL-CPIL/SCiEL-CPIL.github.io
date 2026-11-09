import time

_N = 20
_COLORS = 20
_DEADLINE = 140.0  # safety margin under 150s

def construct():
    start = time.time()
    best = None
    best_count = -1

    def now_ok():
        return (time.time() - start) < _DEADLINE

    def evaluate_add(colmat, adj, u, v, c):
        # returns True if adding edge (u,v) with color c keeps all 4-cycles rainbow
        # new 4-cycles: u-v-w-x-u where w in N(v), x in N(u), edge(w,x) exists
        for w in adj[v]:
            if w == u:
                continue
            cw = colmat[v][w]
            if cw is None:
                continue
            if cw == c:
                # edge v-w same color as u-v: any 4-cycle u-v-w-x would have c,c -> invalid
                # but only if there exists x with edge w-x and u-x; we must check
                pass
            for x in adj[u]:
                if x == v or x == w:
                    continue
                if colmat[w][x] is None:
                    continue
                cx = colmat[u][x]
                cwx = colmat[w][x]
                # 4-cycle u-v-w-x: colors c, cw, cwx, cx
                if len({c, cw, cwx, cx}) != 4:
                    return False
        return True

    def try_build(order):
        colmat = [[None]*_N for _ in range(_N)]
        adj = [set() for _ in range(_N)]
        count = 0
        for (u, v) in order:
            if not now_ok():
                break
            if colmat[u][v] is not None:
                continue
            for c in range(_COLORS):
                if evaluate_add(colmat, adj, u, v, c):
                    colmat[u][v] = c
                    colmat[v][u] = c
                    adj[u].add(v)
                    adj[v].add(u)
                    count += 1
                    break
        edges = []
        for u in range(_N):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v, colmat[u][v]])
        return edges

    # generate all unordered pairs
    pairs = [(u, v) for u in range(_N) for v in range(u+1, _N)]

    # Try several deterministic + random orderings
    import random
    rng = random.Random(12345)

    orderings = []
    # natural order
    orderings.append(list(pairs))
    # random orders
    for _ in range(8):
        if not now_ok():
            break
        o = list(pairs)
        rng.shuffle(o)
        orderings.append(o)

    for order in orderings:
        if not now_ok():
            break
        edges = try_build(order)
        if len(edges) > best_count:
            best_count = len(edges)
            best = edges

    # local improvement: try to add remaining edges with color search
    if best is not None and now_ok():
        # rebuild colmat/adj from best
        colmat = [[None]*_N for _ in range(_N)]
        adj = [set() for _ in range(_N)]
        for (u, v, c) in best:
            colmat[u][v] = c
            colmat[v][u] = c
            adj[u].add(v)
            adj[v].add(u)
        # try to add each missing edge
        for (u, v) in pairs:
            if not now_ok():
                break
            if colmat[u][v] is not None:
                continue
            for c in range(_COLORS):
                if evaluate_add(colmat, adj, u, v, c):
                    colmat[u][v] = c
                    colmat[v][u] = c
                    adj[u].add(v)
                    adj[v].add(u)
                    break
        # rebuild edges
        new_edges = []
        for u in range(_N):
            for v in adj[u]:
                if u < v:
                    new_edges.append([u, v, colmat[u][v]])
        if len(new_edges) > best_count:
            best_count = len(new_edges)
            best = new_edges

    if best is None:
        # fallback: star (always valid)
        best = [[0, v, 0] for v in range(1, _N)]

    return best
