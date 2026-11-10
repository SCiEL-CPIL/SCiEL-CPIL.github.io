import random
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 18

    # Precompute masks for sizes 12 and 11
    subsets_12 = []
    for combo in combinations(range(n), 12):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        subsets_12.append(mask)

    subsets_11 = []
    for combo in combinations(range(n), 11):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        subsets_11.append(mask)

    def is_tf_mask(mask, adj):
        remaining = mask
        while remaining:
            lsb_u = remaining & -remaining
            u = lsb_u.bit_length() - 1
            remaining ^= lsb_u
            nu = adj[u] & mask
            v_mask = nu
            while v_mask:
                lsb_v = v_mask & -v_mask
                v = lsb_v.bit_length() - 1
                v_mask ^= lsb_v
                if adj[u] & adj[v] & mask:
                    return False
        return True

    def max_tf(adj):
        for m in subsets_12:
            if is_tf_mask(m, adj):
                return 12
        for m in subsets_11:
            if is_tf_mask(m, adj):
                return 11
        return 10

    def k4_free(adj):
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    continue
                common = adj[i] & adj[j] & ~((1 << i) | (1 << j))
                cm = common
                while cm:
                    lsb_k = cm & -cm
                    k = lsb_k.bit_length() - 1
                    cm ^= lsb_k
                    rest = common & ~((1 << (k + 1)) - 1)
                    if rest & adj[k]:
                        return False
        return True

    def to_edges(adj):
        return [[u, v] for u in range(n) for v in range(u + 1, n) if adj[u] & (1 << v)]

    # Initial: K6,6,6
    cur = [0] * n
    for i in range(6):
        for j in range(6, 18):
            cur[i] |= (1 << j)
            cur[j] |= (1 << i)
    for i in range(6, 12):
        for j in range(12, 18):
            cur[i] |= (1 << j)
            cur[j] |= (1 << i)

    best = cur[:]
    best_sc = 12

    rng = random.Random(7)
    edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    T = 1.0
    Tmin = 0.005
    cool = 0.9995

    while time.time() < deadline:
        u, v = rng.choice(edges)
        he = bool(cur[u] & (1 << v))
        if he:
            cur[u] &= ~(1 << v)
            cur[v] &= ~(1 << u)
        else:
            cur[u] |= (1 << v)
            cur[v] |= (1 << u)

        if not k4_free(cur):
            if he:
                cur[u] |= (1 << v)
                cur[v] |= (1 << u)
            else:
                cur[u] &= ~(1 << v)
                cur[v] &= ~(1 << u)
            T = max(Tmin, T * cool)
            continue

        sc = max_tf(cur)
        if sc < best_sc:
            best = cur[:]
            best_sc = sc
            if best_sc <= 10:
                break
        elif sc > best_sc:
            if rng.random() >= math.exp(-(sc - best_sc) / T):
                if he:
                    cur[u] |= (1 << v)
                    cur[v] |= (1 << u)
                else:
                    cur[u] &= ~(1 << v)
                    cur[v] &= ~(1 << u)
        T = max(Tmin, T * cool)

    return to_edges(best)
