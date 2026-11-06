"""Maximum-weight independent set in the lcm-conflict graph over [2,256]."""
# EVOLVE-BLOCK-START
import time
import math

_START = time.time()
_DEADLINE = _START + 80.0  # safety margin under the 90s wall-clock kill


def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 80.0

    nodes = list(range(2, 257))

    # Universally compatible: primes in (128, 256]. Always included.
    universal = [p for p in nodes if _is_prime(p) and p > 128]
    universal_set = set(universal)

    search_nodes = [v for v in nodes if v not in universal_set]

    w = {v: 1.0 / v for v in search_nodes}

    # Conflict masks (bit over value positions).
    conflict_mask = {}
    for v in search_nodes:
        m = 0
        for u in search_nodes:
            if u != v and math.lcm(v, u) <= 256:
                m |= (1 << u)
        conflict_mask[v] = m

    # ---- Dominance reduction: drop v if a higher-weight u conflicts with a
    # ---- subset of v's conflicts (u is at least as compatible and as heavy).
    order_w = sorted(search_nodes, key=lambda v: (-w[v], v))
    kept = []
    for v in order_w:
        dominated = False
        mv = conflict_mask[v]
        for u in kept:
            # u earlier => w[u] >= w[v]; need conflict_mask[u] subset of mv
            if (conflict_mask[u] & mv) == conflict_mask[u]:
                dominated = True
                break
        if not dominated:
            kept.append(v)
    search_nodes = kept

    # Rebuild masks restricted to kept (clear bits for removed nodes).
    kept_mask_all = 0
    for v in search_nodes:
        kept_mask_all |= (1 << v)
    for v in search_nodes:
        conflict_mask[v] &= kept_mask_all

    conflict_count = {v: conflict_mask[v].bit_count() for v in search_nodes}

    def argmax_weight(mask):
        best_v = -1
        best_w = -1.0
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            if w[v] > best_w:
                best_w = w[v]
                best_v = v
            m ^= lsb
        return best_v

    def sum_weight(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            s += w[v]
            m ^= lsb
        return s

    def clique_cover_bound(cands):
        # Upper bound: independent set takes <=1 per conflict-clique.
        bound = 0.0
        while cands:
            v = argmax_weight(cands)
            clique = 1 << v
            rem = conflict_mask[v] & cands
            while rem:
                lsb = rem & (-rem)
                u = lsb.bit_length() - 1
                if (conflict_mask[u] & clique) == clique:
                    clique |= lsb
                rem ^= lsb
            bound += w[v]
            cands &= ~clique
        return bound

    # ---- Greedy incumbents (multiple orderings).
    def greedy(order):
        chosen = 0
        wt = 0.0
        for v in order:
            if (conflict_mask[v] & chosen) == 0:
                chosen |= (1 << v)
                wt += w[v]
        return wt, chosen

    orders = [
        sorted(search_nodes, key=lambda v: (-w[v], v)),
        sorted(search_nodes, key=lambda v: (conflict_count[v], -w[v], v)),
        sorted(search_nodes, key=lambda v: (-w[v], conflict_count[v], v)),
        sorted(search_nodes, key=lambda v: (conflict_count[v], v)),
    ]
    best_weight = 0.0
    best_set = 0
    for o in orders:
        wt, chosen = greedy(o)
        if wt > best_weight:
            best_weight = wt
            best_set = chosen

    # ---- Branch and bound with bitsets + deadline guard.
    import sys
    sys.setrecursionlimit(100000)

    def expand(cands, cur_weight, chosen):
        nonlocal best_weight, best_set
        if time.time() > _DEADLINE:
            return
        if cands == 0:
            if cur_weight > best_weight:
                best_weight = cur_weight
                best_set = chosen
            return
        # loose bound first
        if cur_weight + sum_weight(cands) <= best_weight + 1e-12:
            return
        # tight bound
        if cur_weight + clique_cover_bound(cands) <= best_weight + 1e-12:
            return
        v = argmax_weight(cands)
        # include v
        expand(cands & ~conflict_mask[v] & ~(1 << v),
               cur_weight + w[v], chosen | (1 << v))
        if time.time() > _DEADLINE:
            return
        # exclude v
        expand(cands & ~(1 << v), cur_weight, chosen)

    expand(kept_mask_all, 0.0, 0)

    # ---- Local search polish: try adding any compatible node.
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        for v in search_nodes:
            if (best_set >> v) & 1:
                continue
            if (conflict_mask[v] & best_set) == 0:
                best_set |= (1 << v)
                best_weight += w[v]
                improved = True

    # ---- Assemble final set (universal primes + chosen search nodes).
    result = set(universal)
    m = best_set
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        result.add(v)
        m ^= lsb
    return sorted(result)
# EVOLVE-BLOCK-END
