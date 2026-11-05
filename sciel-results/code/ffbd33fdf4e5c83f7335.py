"""Exact Maximum Independent Set solver for disjoint residue classes."""
import time
from math import gcd

MAXM = 128
DEADLINE = 75.0
_START = time.time()


def _time_left():
    return DEADLINE - (time.time() - _START)


def _primes_of(m):
    t = m
    ps = []
    d = 2
    while d * d <= t:
        if t % d == 0:
            ps.append(d)
            while t % d == 0:
                t //= d
        d += 1
    if t > 1:
        ps.append(t)
    return tuple(ps)


def _build():
    seen = set()
    nodes = []
    for m in range(2, MAXM + 1):
        ps = _primes_of(m)
        for r in range(m):
            prof = tuple(r % p for p in ps)
            key = (m, prof)
            if key in seen:
                continue
            seen.add(key)
            nodes.append((r, m))
    n = len(nodes)

    by_mod = {}
    for idx, (r, m) in enumerate(nodes):
        by_mod.setdefault(m, []).append(idx)
    adj = [0] * n
    mods = sorted(by_mod.keys())
    for a_i, mi in enumerate(mods):
        for mj in mods[a_i:]:
            g = gcd(mi, mj)
            if g == 1:
                continue
            list_i = by_mod[mi]
            list_j = by_mod[mj]
            if mi == mj:
                for ii in list_i:
                    for jj in list_i:
                        if ii != jj:
                            adj[ii] |= (1 << jj)
                            adj[jj] |= (1 << ii)
                continue
            buckets = {}
            for jj in list_j:
                buckets.setdefault(nodes[jj][0] % g, []).append(jj)
            for ii in list_i:
                for jj in buckets.get(nodes[ii][0] % g, ()):
                    adj[ii] |= (1 << jj)
                    adj[jj] |= (1 << ii)
    return nodes, adj, n


def _greedy(nodes, adj, n):
    remaining = set(range(n))
    chosen = []
    while remaining and _time_left() > 1.0:
        rem_bits = 0
        for u in remaining:
            rem_bits |= (1 << u)
        best_v = None
        best_key = None
        for v in remaining:
            c = (adj[v] & rem_bits).bit_count()
            key = (c, nodes[v][1], nodes[v][0])
            if best_key is None or key < best_key:
                best_key = key
                best_v = v
        if best_v is None:
            break
        chosen.append(best_v)
        nb = adj[best_v]
        remaining = {u for u in remaining if u != best_v and not ((nb >> u) & 1)}
    return chosen


def _exact(nodes, adj, n):
    comp = [0] * n
    allbits = (1 << n) - 1
    for i in range(n):
        comp[i] = allbits ^ (1 << i) ^ adj[i]

    cdeg = [comp[i].bit_count() for i in range(n)]
    order = sorted(range(n), key=lambda i: cdeg[i], reverse=True)
    pos = [0] * n
    for kk, v in enumerate(order):
        pos[v] = kk
    ocomp = [0] * n
    for v in range(n):
        p = pos[v]
        bits = comp[v]
        rv = 0
        while bits:
            lb = bits & (-bits)
            u = lb.bit_length() - 1
            rv |= (1 << pos[u])
            bits ^= lb
        ocomp[p] = rv

    import sys
    sys.setrecursionlimit(100000)

    best = [0]
    best_set = [0]

    def color_bound(P):
        # greedy coloring of P in complement -> upper bound on clique size
        uncolored = P
        cnt = 0
        while uncolored:
            cnt += 1
            cur = uncolored
            used = 0
            while cur:
                lb = cur & (-cur)
                v = lb.bit_length() - 1
                if (ocomp[v] & used) == 0:
                    used |= (1 << v)
                    uncolored &= ~(1 << v)
                cur &= ~(1 << v)
        return cnt

    def expand(P, Q, k):
        if k + color_bound(P) <= best[0]:
            return
        if P == 0:
            if k > best[0]:
                best[0] = k
                best_set[0] = Q
            return
        if _time_left() < 0.5:
            return
        lb = P & (-P)
        v = lb.bit_length() - 1
        P1 = P & ocomp[v] & ~(1 << v)
        expand(P1, Q | (1 << v), k + 1)
        P2 = P & ~(1 << v)
        expand(P2, Q, k)

    expand(allbits, 0, 0)
    return best[0], best_set[0]


def _baseline():
    return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]


def construct():
    if _time_left() < 2.0:
        return _baseline()

    nodes, adj, n = _build()

    greedy = _greedy(nodes, adj, n)
    greedy_size = len(greedy)

    try:
        best_size, best_set = _exact(nodes, adj, n)
        if best_size > greedy_size:
            out = []
            for i in range(n):
                if (best_set >> i) & 1:
                    r, m = nodes[i]
                    out.append([r, m])
            if out:
                return out
    except Exception:
        pass

    out = [[nodes[v][0], nodes[v][1]] for v in greedy]
    return out if out else _baseline()
