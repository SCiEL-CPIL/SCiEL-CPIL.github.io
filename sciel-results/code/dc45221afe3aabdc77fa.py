import time
import random

_N = 40
_ADJ_SIZE = 1 << 40

def construct():
    deadline = time.time() + 145.0
    rng = random.Random(123456789)

    # ---------- fast C6 detection using bit masks ----------
    def has_c6(adj):
        # adj: list of int bitmasks
        # For each edge (a,b), check for a 4-path a-x-y-z-b with x!=b, z!=a.
        for a in range(_N):
            ab = adj[a]
            b = ab & -ab
            while b:
                bi = b.bit_length() - 1
                b &= b - 1
                # common neighbors of a and b
                common = ab & adj[bi]
                # exclude a and bi themselves
                common &= ~((1 << a) | (1 << bi))
                while common:
                    x = common & -common
                    common ^= x
                    xi = x.bit_length() - 1
                    # z in adj[x] & adj[bi], z != a, z != bi
                    zmask = adj[xi] & adj[bi]
                    zmask &= ~((1 << a) | (1 << bi))
                    if zmask:
                        z = zmask & -zmask
                        zi = z.bit_length() - 1
                        # e in adj[z] & adj[bi], e != a, e != bi, e != z
                        emask = adj[zi] & adj[bi]
                        emask &= ~((1 << a) | (1 << bi) | z)
                        if emask:
                            return True
        return False

    def edge_count(adj):
        c = 0
        for a in range(_N):
            c += adj[a].bit_count()
        return c // 2

    # ---------- greedy build from random ordering ----------
    def greedy_build(order):
        adj = [0] * _N
        for a in order:
            cand = [b for b in range(_N) if b > a and not (adj[a] >> b) & 1]
            for b in cand:
                adj[a] |= 1 << b
                adj[b] |= 1 << a
                if has_c6(adj):
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
        return adj

    def addable_edges(adj):
        out = []
        for a in range(_N):
            missing = ~adj[a] & ((1 << _N) - 1)
            missing &= ~((1 << a))
            missing &= ((1 << (a + 1)) - 1) ^ ((1 << a) - 1)  # only b > a
            m = missing
            while m:
                b = m & -m
                m ^= b
                bi = b.bit_length() - 1
                out.append((a, bi))
        return out

    def polish(adj):
        # repeatedly try to add every missing edge until none can be added
        progress = True
        while progress:
            progress = False
            for a in range(_N):
                for b in range(a + 1, _N):
                    if (adj[a] >> b) & 1:
                        continue
                    adj[a] |= 1 << b
                    adj[b] |= 1 << a
                    if has_c6(adj):
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                    else:
                        progress = True
        return adj

    def adj_to_edges(adj):
        edges = []
        for a in range(_N):
            for b in range(a + 1, _N):
                if (adj[a] >> b) & 1:
                    edges.append([a, b])
        return edges

    # ---------- initial structured start: 8 copies of K5 ----------
    def initial_adj():
        adj = [0] * _N
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    adj[base + a] |= 1 << (base + b)
                    adj[base + b] |= 1 << (base + a)
        return adj

    best_adj = initial_adj()
    best_cnt = edge_count(best_adj)
    best_adj = polish(best_adj)
    best_cnt = edge_count(best_adj)

    # ---------- greedy restarts ----------
    for it in range(2000):
        if time.time() > deadline - 5.0:
            break
        order = list(range(_N))
        rng.shuffle(order)
        adj = greedy_build(order)
        adj = polish(adj)
        cnt = edge_count(adj)
        if cnt > best_cnt:
            best_cnt = cnt
            best_adj = adj
            if best_cnt >= 105:
                break

    # ---------- final exhaustive polish on the best ----------
    if time.time() < deadline - 2.0:
        best_adj = polish(best_adj)
        best_cnt = edge_count(best_adj)

    return adj_to_edges(best_adj)
