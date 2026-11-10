import time
from itertools import combinations

N = 20
DEADLINE = 115.0  # safety margin before 120s kill


def construct():
    start = time.time()
    t0 = start
    deadline = t0 + DEADLINE

    # Precompute all vertex pairs (potential edges)
    pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    M = len(pairs)  # 190
    pair_idx = {}
    for i, (u, v) in enumerate(pairs):
        pair_idx[(u, v)] = i
        pair_idx[(v, u)] = i

    # Bitsets: neighbor-of-vertex and incident-to-vertex
    neigh = [0] * N
    incident = [0] * N
    for i, (u, v) in enumerate(pairs):
        bit = 1 << i
        neigh[u] |= bit
        neigh[v] |= bit
        incident[u] |= bit
        incident[v] |= bit

    # conflict[e] = bitset of other edges that together with e form a K3,3
    conflict = [0] * M
    for (u, v) in combinations(range(N), 2):
        nu = neigh[u] & incident[u]
        nv = neigh[v] & incident[v]
        common = nu & nv  # edges incident to both u and v (K2,2)
        for (u2, v2) in combinations(range(N), 2):
            if u2 == u or v2 == v or u2 == v or v2 == u:
                continue
            nu2 = neigh[u2] & incident[u2]
            nv2 = neigh[v2] & incident[v2]
            common2 = nu2 & nv2
            inter = common & common2
            if (inter & (inter - 1)) != 0:  # >=2 common edges
                i1 = pair_idx[(u, v)]
                i2 = pair_idx[(u2, v2)]
                conflict[i1] |= 1 << i2
                conflict[i2] |= 1 << i1

    # Greedy seed: add edges in order of fewest conflicts (least restrictive)
    order = sorted(range(M), key=lambda i: conflict[i].bit_count())
    used = 0
    for i in order:
        if (used & (1 << i)) == 0 and (used & conflict[i]) == 0:
            used |= 1 << i
    best = used

    # Branch-and-bound with bitset upper bound
    best_ref = [best]
    nodes = [0]
    tlast = [t0]

    def bound(used, avail):
        # greedy packing upper bound on additional edges
        a = avail
        c = 0
        while a:
            lb = a & (-a)
            c += 1
            a &= ~(lb | conflict[lb.bit_length() - 1])
        return c

    def dfs(used, avail, depth):
        nonlocal deadline
        if time.time() - tlast[0] > 1.0:
            tlast[0] = time.time()
            if time.time() > deadline:
                return
        nodes[0] += 1
        if nodes[0] % 2000 == 0 and time.time() > deadline:
            return
        if avail == 0:
            return
        cur = used.bit_count() + avail.bit_count()
        if cur <= best_ref[0].bit_count():
            return
        ub = used.bit_count() + bound(used, avail)
        if ub <= best_ref[0].bit_count():
            return
        lb = avail & (-avail)
        i = lb.bit_length() - 1
        new_used = used | lb
        if (new_used & conflict[i]) == 0:
            new_avail = avail & ~lb & ~conflict[i]
            if new_used.bit_count() + bound(new_used, new_avail) > best_ref[0].bit_count():
                if new_used.bit_count() > best_ref[0].bit_count():
                    best_ref[0] = new_used
                dfs(new_used, new_avail, depth + 1)
        # branch exclude i
        dfs(used, avail & ~lb, depth + 1)

    dfs(best, (1 << M) - 1, 0)
    best = best_ref[0]

    edges = []
    for i in range(M):
        if (best >> i) & 1:
            u, v = pairs[i]
            edges.append([u, v])
    return edges
