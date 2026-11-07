import random
import time

_N = 15
_NP = _N * (_N - 1) // 2
_MASK = (1 << _NP) - 1
_DEADLINE = 85.0  # safety margin under 90s


def _idx(u, v):
    if u > v:
        u, v = v, u
    return u * _N - u * (u + 1) // 2 - v


def _build_tables():
    idx = [[0] * _N for _ in range(_N)]
    adj = [set() for _ in range(_N)]
    neigh = [[] for _ in range(_N)]
    for u in range(_N):
        for v in range(u + 1, _N):
            i = _idx(u, v)
            idx[u][v] = i
            idx[v][u] = i
            adj[u].add(v)
            adj[v].add(u)
            neigh[u].append(v)
            neigh[v].append(u)
    return idx, adj, neigh


IDX, ADJ, NEIGH = _build_tables()


def _deg(mask, i):
    return bin(mask & ADJ[i]).count('1')


def _bad(mask):
    # returns True if some simple 3-edge path has equal-degree endpoints
    degs = [_deg(mask, i) for i in range(_N)]
    for u in range(_N):
        du = degs[u]
        for v in NEIGH[u]:
            if not (mask & (1 << IDX[u][v])):
                continue
            dv = degs[v]
            for s in NEIGH[u]:
                if s == v or not (mask & (1 << IDX[u][s])):
                    continue
                ds = degs[s]
                if ds != dv:
                    continue
                for t in NEIGH[v]:
                    if t == s or t == u:
                        continue
                    if (mask & (1 << IDX[v][t])) and degs[t] == ds:
                        return True
    return False


def _valid(mask):
    return not _bad(mask)


def _greedy_start(rng):
    order = list(range(_NP))
    rng.shuffle(order)
    mask = 0
    for i in order:
        m2 = mask | (1 << i)
        if _valid(m2):
            mask = m2
    return mask


def _neighbors_of_edge(mask, i):
    u, v = divmod(i, _N)
    if u > v:
        u, v = v, u
    res = []
    for s in NEIGH[u]:
        if s != v and (mask & (1 << IDX[u][s])):
            for t in NEIGH[v]:
                if t != s and t != u and (mask & (1 << IDX[v][t])):
                    res.append((s, t))
    return res


def _local_search(mask, rng, deadline):
    n = bin(mask).count('1')
    while time.time() < deadline:
        i = rng.randrange(_NP)
        if mask & (1 << i):
            m2 = mask & ~(1 << i)
            if _valid(m2):
                mask = m2
                n -= 1
        else:
            m2 = mask | (1 << i)
            if _valid(m2):
                mask = m2
                n += 1
    return mask, n


def construct():
    deadline = time.time() + _DEADLINE
    best = 0
    best_mask = 0

    # deterministic-ish first start
    rng = random.Random(12345)
    m0 = _greedy_start(rng)
    n0 = bin(m0).count('1')
    if n0 > best:
        best, best_mask = n0, m0

    # multi-start local search
    seed = 0
    while time.time() < deadline:
        seed += 1
        r = random.Random(seed)
        m = _greedy_start(r)
        m, n = _local_search(m, r, deadline)
        if n > best:
            best, best_mask = n, m
        if best >= 45:
            break

    edges = []
    for i in range(_NP):
        if best_mask & (1 << i):
            u, v = divmod(i, _N)
            if u > v:
                u, v = v, u
            edges.append([int(u), int(v)])
    return edges
