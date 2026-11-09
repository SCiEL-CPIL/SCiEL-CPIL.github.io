# EVOLVE-BLOCK-START
import time
from random import Random

_TIME_LIMIT = 140.0
_DEADLINE = None

def _now():
    return time.time()

def construct():
    global _DEADLINE
    _DEADLINE = _now() + _TIME_LIMIT
    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if _now() > _DEADLINE - 6:
            out.append(_fast_greedy(*spec))
            continue
        out.append(_solve_instance(*spec))
    return out

def _instance(n, d, seed):
    rng = Random(seed)
    degree, edges = [0] * n, set()
    for _ in range(4 * n * d):
        u, v = rng.randrange(n), rng.randrange(n)
        e = (min(u, v), max(u, v))
        if u != v and degree[u] < d and degree[v] < d and e not in edges:
            edges.add(e)
            degree[u] += 1
            degree[v] += 1
    return sorted(edges)

def _build_conflict(n, edges):
    m = len(edges)
    adj = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    eidx = {e: i for i, e in enumerate(edges)}
    conflict = [0] * m
    for i, (u, v) in enumerate(edges):
        mask = 0
        near = {u, v}
        for x in (u, v):
            near |= adj.get(x, set())
        for y in near:
            for z in adj.get(y, ()):
                if z > y:
                    continue
                e2 = (min(y, z), max(y, z))
                if e2 in eidx and e2 != (u, v):
                    j = eidx[e2]
                    mask |= (1 << j)
        conflict[i] = mask
    return conflict

def _greedy_color(conflict, order):
    m = len(order)
    colour = [-1] * m
    used = 0
    for i in order:
        c = 0
        occ = set()
        mm = conflict[i]
        while mm:
            lsb = mm & (-mm)
            j = lsb.bit_length() - 1
            if colour[j] >= 0:
                occ.add(colour[j])
            mm -= lsb
        while c in occ:
            c += 1
        colour[i] = c
        if c + 1 > used:
            used = c + 1
    return colour, used

def _fast_greedy(n, d, seed):
    edges = _instance(n, d, seed)
    conflict = _build_conflict(n, edges)
    m = len(edges)
    order = list(range(m))
    colour, used = _greedy_color(conflict, order)
    return [colour[i] for i in range(m)]

def _count(colour):
    return len(set(c for c in colour if c >= 0))

def _local_search(conflict, colour, rng, max_iters=20000):
    m = len(conflict)
    best = _count(colour)
    best_colour = colour[:]
    improved = True
    it = 0
    while improved and it < max_iters:
        improved = False
        it += 1
        groups = {}
        for i, c in enumerate(colour):
            if c >= 0:
                groups.setdefault(c, 0)
                groups[c] |= (1 << i)
        present = sorted(groups.keys(), reverse=True)
        for a in present:
            for b in present:
                if a <= b:
                    continue
                if conflict_mask_intersect(groups[a], groups[b], conflict):
                    continue
                # merge a into b
                for i in range(m):
                    if (groups[a] >> i) & 1:
                        colour[i] = b
                del groups[a]
                improved = True
                if _count(colour) < best:
                    best = _count(colour)
                    best_colour = colour[:]
                break
            if improved:
                break
    return best, best_colour

def _conflict_mask_intersect(maskA, maskB, conflict):
    # check if any edge in A conflicts with any edge in B
    mm = maskA
    while mm:
        lsb = mm & (-mm)
        i = lsb.bit_length() - 1
        if conflict[i] & maskB:
            return True
        mm -= lsb
    return False

def _solve_instance(n, d, seed):
    edges = _instance(n, d, seed)
    conflict = _build_conflict(n, edges)
    m = len(edges)
    best = m
    best_colour = None
    rng = Random(seed * 1000 + 7)

    orderings = []
    orderings.append(list(range(m)))
    deg = [conflict[i].bit_count() for i in range(m)]
    orderings.append(sorted(range(m), key=lambda i: (-deg[i], i)))
    orderings.append(sorted(range(m), key=lambda i: (deg[i], i)))
    for _ in range(40):
        o = list(range(m))
        rng.shuffle(o)
        orderings.append(o)

    for o in orderings:
        if _now() > _DEADLINE - 10:
            break
        colour, used = _greedy_color(conflict, o)
        if used < best:
            best = used
            best_colour = colour[:]
        b2, c2 = _local_search(conflict, colour[:], rng, max_iters=3000)
        if b2 < best:
            best = b2
            best_colour = c2[:]

    if best_colour is None:
        best_colour = [0] * m
    return [best_colour[i] for i in range(m)]
# EVOLVE-BLOCK-END
