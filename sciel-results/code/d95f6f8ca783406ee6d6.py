import time, random, math

def construct():
    N = 16
    START = time.time()
    DEADLINE = START + 140.0          # safety margin under the 150 s kill
    rng = random.Random(20240517)

    def eval_adj(adj):
        deg = [adj[i].bit_count() for i in range(N)]
        maxsum = 0
        total = 0
        worst = None
        for u in range(N):
            au = adj[u]
            du = deg[u]
            for v in range(u + 1, N):
                if (au >> v) & 1:
                    common = au & adj[v] & (~((1 << (v + 1)) - 1))  # w > v
                    w = common
                    dv = deg[v]
                    while w:
                        lsb = w & (-w)
                        ww = lsb.bit_length() - 1
                        s = du + dv + deg[ww]
                        total += s
                        if s > maxsum:
                            maxsum = s
                            worst = (u, v, ww)
                        w ^= lsb
        return maxsum, total, worst, deg

    def obj(maxsum, total):
        # max dominates (1000/unit), total is a smooth tie-breaker
        return maxsum * 1000 + total

    def edges_of(adj):
        el = []
        for u in range(N):
            for v in range(u + 1, N):
                if (adj[u] >> v) & 1:
                    el.append((u, v))
        return el

    def nonedges_of(adj, excl):
        nl = []
        for u in range(N):
            for v in range(u + 1, N):
                if ((adj[u] >> v) & 1) == 0 and (u, v) != excl:
                    nl.append((u, v))
        return nl

    def init_k88():
        adj = [0] * N
        for u in range(8):
            for v in range(8, 16):
                adj[u] |= 1 << v
                adj[v] |= 1 << u
        adj[0] |= 1 << 1
        adj[1] |= 1 << 0
        return adj

    def init_random():
        pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
        rng.shuffle(pairs)
        adj = [0] * N
        for u, v in pairs[:65]:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        return adj

    best_adj = None
    best_max = 10 ** 9
    best_total = 10 ** 9

    def consider(adj):
        nonlocal best_adj, best_max, best_total
        m, t, w, d = eval_adj(adj)
        if m < best_max or (m == best_max and t < best_total):
            best_max = m
            best_total = t
            best_adj = adj[:]
        return m, t

    consider(init_k88())

    def anneal(adj, budget):
        m, t, w, d = eval_adj(adj)
        cur_o = obj(m, t)
        T = 2000.0
        Tmin = 1.0
        cool = 0.99995
        end = time.time() + budget
        while time.time() < end:
            el = edges_of(adj)
            if w is not None and rng.random() < 0.3:
                # bias removal toward an edge of the current worst triangle
                u, v = rng.choice([(w[0], w[1]), (w[0], w[2]), (w[1], w[2])])
            else:
                u, v = rng.choice(el)
            nl = nonedges_of(adj, (u, v))
            if not nl:
                break
            x, y = rng.choice(nl)
            # apply swap: remove (u,v), add (x,y)  -> edge count stays 65
            adj[u] ^= 1 << v
            adj[v] ^= 1 << u
            adj[x] |= 1 << y
            adj[y] |= 1 << x
            nm, nt, nw, nd = eval_adj(adj)
            no = obj(nm, nt)
            delta = no - cur_o
            if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                m, t, w = nm, nt, nw
                cur_o = no
                consider(adj)
            else:
                # revert
                adj[x] ^= 1 << y
                adj[y] ^= 1 << x
                adj[u] |= 1 << v
                adj[v] |= 1 << u
            T *= cool
            if T < Tmin:
                T = Tmin
        return m, t

    attempt = 1
    while time.time() < DEADLINE - 5.0:
        attempt += 1
        if attempt % 5 == 1:
            adj = init_k88()
        else:
            adj = init_random()
        anneal(adj, 4.0)
        if best_max <= 24:
            break

    # strict hill-climb polish on the best graph found
    if best_adj is not None:
        adj = best_adj[:]
        m, t, w, d = eval_adj(adj)
        cur_o = obj(m, t)
        for _ in range(30000):
            if time.time() > DEADLINE - 2.0:
                break
            el = edges_of(adj)
            u, v = rng.choice(el)
            nl = nonedges_of(adj, (u, v))
            if not nl:
                break
            x, y = rng.choice(nl)
            adj[u] ^= 1 << v
            adj[v] ^= 1 << u
            adj[x] |= 1 << y
            adj[y] |= 1 << x
            nm, nt, nw, nd = eval_adj(adj)
            if obj(nm, nt) < cur_o:
                m, t, w = nm, nt, nw
                cur_o = obj(nm, nt)
                consider(adj)
            else:
                adj[x] ^= 1 << y
                adj[y] ^= 1 << x
                adj[u] |= 1 << v
                adj[v] |= 1 << u

    if best_adj is None:
        best_adj = init_k88()
    edges = [[u, v] for (u, v) in edges_of(best_adj)]
    return {'n': 16, 'edges': edges}
