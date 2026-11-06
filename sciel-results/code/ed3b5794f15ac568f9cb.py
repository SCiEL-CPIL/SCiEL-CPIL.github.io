import time
from math import lcm
import random

def construct():
    t0 = time.time()
    TIME_LIMIT = 54.0
    vals = list(range(1, 129))
    n = 128
    full = (1 << n) - 1

    # Precompute pairwise LCM matrix.
    L = [[0] * n for _ in range(n)]
    for i in range(n):
        vi = vals[i]
        row = L[i]
        for j in range(i + 1, n):
            v = lcm(vi, vals[j])
            row[j] = v
            L[j][i] = v

    # comp[i][j] = bitset of k such that {i,j,k} is a bad triple
    # (lcm(i,j)==lcm(i,k)==lcm(j,k)).  Also count hypergraph degrees.
    comp = [[0] * n for _ in range(n)]
    edge_count = [0] * n
    for i in range(n):
        Li = L[i]
        ci = comp[i]
        for j in range(i + 1, n):
            lv = Li[j]
            Lj = L[j]
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                if Li[k] == lv and Lj[k] == lv:
                    mask |= 1 << k
            ci[j] = mask
            comp[j][i] = mask
            mm = mask
            while mm:
                k = (mm & -mm).bit_length() - 1
                mm ^= 1 << k
                edge_count[i] += 1
                edge_count[j] += 1
                edge_count[k] += 1

    # Greedy (order-driven) independent-set builder.
    def greedy(order):
        S = []
        F = 0
        for v in order:
            if (1 << v) & F:
                continue
            S.append(v)
            nf = F
            for a in S[:-1]:
                nf |= comp[v][a]
            F = nf
        return S

    rng = random.Random(12345)
    base = list(range(n))
    best_len = 0
    best_set = []
    for _ in range(400):
        order = base[:]
        rng.shuffle(order)
        S = greedy(order)
        if len(S) > best_len:
            best_len = len(S)
            best_set = S[:]
        if time.time() - t0 > TIME_LIMIT * 0.25:
            break
    if not best_set:
        best_set = greedy(base)
        best_len = len(best_set)

    # Greedy matching on a vertex bitset -> lower bound on vertex cover,
    # hence an upper bound on the independent-set size of that induced hypergraph.
    def greedy_matching(R):
        used = 0
        count = 0
        r = R
        while r:
            v = (r & -r).bit_length() - 1
            r ^= 1 << v
            rem = R & ~used & ~(1 << v)
            ra = rem
            while ra:
                a = (ra & -ra).bit_length() - 1
                ra ^= 1 << a
                bmask = comp[a][v] & rem & ~(1 << a) & ~(1 << v)
                if bmask:
                    b = (bmask & -bmask).bit_length() - 1
                    count += 1
                    used |= (1 << v) | (1 << a) | (1 << b)
                    break
        return count

    best = [best_len]
    best_sol = best_set[:]
    nodes = [0]

    def bb(S, F, R):
        nodes[0] += 1
        if time.time() - t0 > TIME_LIMIT:
            return
        avail = R & ~F
        if avail == 0:
            if len(S) > best[0]:
                best[0] = len(S)
                best_sol[:] = S[:]
            return
        # Cheap bound first.
        if len(S) + avail.bit_count() <= best[0]:
            return
        # Stronger bound via a matching in the induced hypergraph on avail.
        m = greedy_matching(avail)
        ub = len(S) + avail.bit_count() - m
        if ub <= best[0]:
            return
        # Branch on the highest-degree available vertex.
        av = avail
        bestv = -1
        bestd = -1
        while av:
            v = (av & -av).bit_length() - 1
            av ^= 1 << v
            d = edge_count[v]
            if d > bestd:
                bestd = d
                bestv = v
        v = bestv
        # Include branch.
        nf = F
        for a in S:
            nf |= comp[v][a]
        S.append(v)
        bb(S, nf, R & ~(1 << v))
        S.pop()
        # Exclude branch.
        bb(S, F, R & ~(1 << v))

    bb([], 0, full)

    res = sorted(vals[i] for i in best_sol)
    if not res:
        res = [1]
    return res
