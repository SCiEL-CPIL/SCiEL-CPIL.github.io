import time
import random
import math
from collections import deque

def construct(k, g):
    t0 = time.time()
    DEADLINE = t0 + 138.0

    def rem():
        return DEADLINE - time.time()

    # ---------------- fast girth (non-backtracking, early exit) ----------------
    def girth(n, adj, g):
        # adj: list of lists (degree k)
        minc = float('inf')
        for s in range(n):
            if time.time() > DEADLINE - 2:
                return -1
            # state = (vertex, prev) ; dist
            dist = {}
            dq = deque()
            for nb in adj[s]:
                dq.append((s, nb))
                dist[(s, nb)] = 1
            while dq:
                u, v = dq.popleft()
                d = dist[(u, v)]
                for w in adj[v]:
                    if w == u:
                        continue
                    st = (v, w)
                    if st in dist:
                        cyc = d + dist[st]
                        if cyc < minc:
                            minc = cyc
                        if cyc < g:
                            return minc
                    else:
                        dist[st] = d + 1
                        dq.append(st)
            if minc < g:
                return minc
        return minc

    # ---------------- Configuration Model random cubic graph ----------------
    def random_cubic(n, rng):
        if n % 2 != 0:
            n += 1
        # half-edges: each vertex appears k times
        pool = []
        for v in range(n):
            for _ in range(k):
                pool.append(v)
        rng.shuffle(pool)
        adj = [[] for _ in range(n)]
        for i in range(0, n * k, 2):
            a = pool[i]; b = pool[i + 1]
            if a == b:
                continue
            if b in adj[a]:
                continue
            adj[a].append(b); adj[b].append(a)
        # check 3-regular & simple
        for v in range(n):
            if len(adj[v]) != k or len(set(adj[v])) != k:
                return None
        return adj

    # ---------------- 2-opt swap to remove short cycles ----------------
    def swap_edges(adj, a, b, c, d, rng):
        # edges (a,b) and (c,d) -> (a,c) and (b,d) or (a,d),(b,c)
        # remove
        adj[a].remove(b); adj[b].remove(a)
        adj[c].remove(d); adj[d].remove(c)
        # add one of the two rewirings
        if rng.random() < 0.5:
            adj[a].append(c); adj[c].append(a)
            adj[b].append(d); adj[d].append(b)
        else:
            adj[a].append(d); adj[d].append(a)
            adj[b].append(c); adj[c].append(b)

    def find_short_cycle(adj, n, target):
        # BFS non-backtracking from a random vertex; return list of vertices on a cycle < target
        # We do a bounded search to find one short cycle quickly.
        s = random.randrange(n)
        dist = {}
        parent = {}
        dq = deque()
        for nb in adj[s]:
            dq.append((s, nb)); dist[(s, nb)] = 1; parent[(s, nb)] = None
        found = None
        while dq and found is None:
            u, v = dq.popleft()
            d = dist[(u, v)]
            for w in adj[v]:
                if w == u:
                    continue
                st = (v, w)
                if st in dist:
                    cyc = d + dist[st]
                    if cyc < target:
                        # reconstruct cycle: forward s..(u,v) and s..(v,w)
                        # forward path from s to v via u
                        path1 = []
                        cur = (u, v)
                        while cur is not None:
                            path1.append(cur[1])
                            cur = parent[cur]
                        # path1 = [v, u, ..., s]; reverse to [s,...,u,v]
                        path1.reverse()
                        # forward path from s to w via v
                        path2 = []
                        cur2 = (v, w)
                        while cur2 is not None:
                            path2.append(cur2[1])
                            cur2 = parent[cur2]
                        path2.reverse()  # [s,...,v,w]
                        # cycle = path1 (s..v) + path2[1:] (..w) but they share s and v
                        # Build unique cycle vertices
                        cycverts = list(path1)
                        # path2 without s (first) -> [..,v,w]; append those after s
                        for x in path2[1:]:
                            if x not in cycverts:
                                cycverts.append(x)
                        found = cycverts
                        break
                else:
                    dist[st] = d + 1
                    parent[st] = (u, v)
                    dq.append(st)
        return found

    def try_swap_on_cycle(adj, cyc, rng):
        # try to break cycle by swapping an edge of the cycle with a non-cycle edge
        # pick a cycle edge (x,y); find a non-neighbor z of x and w of y not in cycle to swap
        L = len(cyc)
        # build set of cycle vertices and cycle edges
        cset = set(cyc)
        cycle_edges = set()
        for i in range(L):
            a = cyc[i]; b = cyc[(i + 1) % L]
            cycle_edges.add((min(a, b), max(a, b)))
        for i in range(L):
            x = cyc[i]; y = cyc[(i + 1) % L]
            # candidates z neighbor of x not in cycle, w neighbor of y not in cycle
            for z in adj[x]:
                if z in cset:
                    continue
                for w in adj[y]:
                    if w in cset:
                        continue
                    # swap (x,y) & (z,w) -> (x,z),(y,w) or (x,w),(y,z)
                    # ensure no multi-edge / loop
                    ok = True
                    if z == y or w == x:
                        ok = False
                    if (min(x, z), max(x, z)) in cycle_edges or (min(y, w), max(y, w)) in cycle_edges:
                        pass
                    # do swap (x,z),(y,w)
                    adj[x].remove(y); adj[y].remove(x)
                    adj[z].remove(w); adj[w].remove(z)
                    # check simple
                    if z in adj[x] or w in adj[y]:
                        # revert
                        adj[x].append(y); adj[y].append(x)
                        adj[z].append(w); adj[w].append(z)
                        continue
                    adj[x].append(z); adj[z].append(x)
                    adj[y].append(w); adj[w].append(y)
                    return True
        return False

    def improve_girth(adj, n, target, budget):
        rng = random.Random(12345)
        end = time.time() + budget
        stall = 0
        while time.time() < end:
            cur = girth(n, adj, target)
            if cur == -1:
                break
            if cur >= target:
                return cur
            # find a short cycle
            cyc = find_short_cycle(adj, n, cur + 1)
            if cyc is None:
                # try random swap
                for _ in range(50):
                    a = rng.randrange(n)
                    b = adj[a][rng.randrange(k)]
                    c = rng.randrange(n)
                    if c == a or c == b:
                        continue
                    d = adj[c][rng.randrange(k)]
                    if d in (a, b) or d == c:
                        continue
                    if b in adj[c] or d in adj[a]:
                        continue
                    swap_edges(adj, a, b, c, d, rng)
                stall += 1
                if stall > 200:
                    break
                continue
            if try_swap_on_cycle(adj, cyc, rng):
                stall = 0
            else:
                stall += 1
                if stall > 200:
                    break
        return girth(n, adj, target)

    # ---------------- LUW fallback (guaranteed valid) ----------------
    def _luw_ops(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        return ((lambda x, y: (x + y) % q),
                (lambda x, y: (x - y) % q),
                (lambda x, y: x * y % q))

    def generate_rules(d, variant=0, offset=0):
        base = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        if variant == 0:
            for c in range(5, d + 1):
                r, c0 = ((c - 5 + offset) % 4), c - ((c - 5 + offset) % 4)
                base[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        elif variant == 1:
            for c in range(5, d + 1):
                base[c] = (1, c - 2) if (c + offset) % 2 == 0 else (c - 1, 1)
        elif variant == 2:
            for c in range(5, d + 1):
                if (c + offset) % 3 == 0:
                    base[c] = (1, c - 3)
                elif (c + offset) % 3 == 1:
                    base[c] = (c - 1, 1)
                else:
                    base[c] = (c - 2, 1)
        else:
            for c in range(5, d + 1):
                r, c0 = ((c - 5 + offset) % 4), c - ((c - 5 + offset) % 4)
                base[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return base

    def build_luw(q, d, rule, time_limit):
        add, sub, mul = _luw_ops(q)

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            if time.time() > time_limit:
                return None
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def luw_construct(kk, gg):
        q = kk
        d = max(3, (gg - 2) // 2)
        best = None
        best_n = float('inf')
        for variant in range(4):
            for offset in range(6):
                if rem() < 12:
                    break
                rule = generate_rules(d, variant, offset)
                res = build_luw(q, d, rule, DEADLINE - 12)
                if res is None:
                    continue
                n, edges = res
                if n >= best_n:
                    continue
                gc = girth(n, [ [j for i,j in edges if i==v] + [i for i,j in edges if j==v] for v in range(n) ], gg) if False else None
                # build adj
                adj = [[] for _ in range(n)]
                for i, j in edges:
                    adj[i].append(j); adj[j].append(i)
                gc = girth(n, adj, gg)
                if gc == -1:
                    continue
                if gc >= gg:
                    best, best_n = (n, edges), n
                    break
            if best is not None:
                break
        if best is None:
            rule = generate_rules(d, 0, 0)
            res = build_luw(q, d, rule, DEADLINE - 2)
            if res is not None:
                best = res
        return best

    # ---------------- main ----------------
    if k != 3:
        res = luw_construct(k, g)
        return res if res is not None else (0, [])

    # k == 3
    rng = random.Random(777)
    # target n: start a bit above the Moore-ish lower estimate for girth g
    # girth 16 -> need ~ 2*(k-1)^(g/2) = 2*2^8 = 512 lower bound; random needs more.
    base_n = 2 * (k - 1) ** (g // 2)
    start_n = max(base_n * 4, 4096)

    best = None
    best_n = float('inf')

    for attempt in range(8):
        if rem() < 25:
            break
        n = start_n + attempt * 1024
        # ensure even
        if n % 2 != 0:
            n += 1
        adj = None
        for _ in range(20):
            if rem() < 20:
                break
            adj = random_cubic(n, rng)
            if adj is not None:
                break
        if adj is None:
            continue
        # improve girth
        budget = max(2.0, rem() - 15)
        cur = improve_girth(adj, n, g, budget)
        if cur == -1:
            continue
        if cur >= g:
            # build edges
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            if n < best_n:
                best = (n, edges)
                best_n = n
            break

    if best is not None:
        return best

    # fallback LUW
    res = luw_construct(k, g)
    return res if res is not None else (0, [])
