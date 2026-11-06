# EVOLVE-BLOCK-START

import time
import random

_N = 20
_ALL_PAIRS = [(i, j) for i in range(_N) for j in range(i + 1, _N)]
# Precompute the set of all triples (as bitmask) so a triple's common-neighbor
# count can be read from a single precomputed lookup table.
_ALL_TRIPLES = []
for _a in range(_N):
    for _b in range(_a + 1, _N):
        for _c in range(_b + 1, _N):
            _ALL_TRIPLES.append((1 << _a) | (1 << _b) | (1 << _c))
_NTRIPLES = len(_ALL_TRIPLES)


def _rebuild_counts(adj):
    """Build cnt[triple] = number of common neighbors outside the triple.
    A graph is K3,3-free iff no cnt entry is >= 3."""
    n = _N
    cnt = [0] * _NTRIPLES
    for i in range(n):
        ai = adj[i]
        for j in range(i + 1, n):
            if (ai >> j) & 1:
                # vertices that are neighbors of both i and j, excluding i and j
                m = ai & adj[j]
                m &= ~(1 << i)
                m &= ~(1 << j)
                # iterate over bits of m
                mm = m
                while mm:
                    lsb = mm & (-mm)
                    k = lsb.bit_length() - 1
                    mm ^= lsb
                    aij = ai & adj[j] & adj[k]
                    aij &= ~(1 << i)
                    aij &= ~(1 << j)
                    aij &= ~(1 << k)
                    c = aij.bit_count()
                    if c:
                        t = (1 << i) | (1 << j) | (1 << k)
                        idx = _TRIPLE_INDEX[t]
                        cnt[idx] = c
    return cnt


# Build a lookup from triple-bitmask -> index
_TRIPLE_INDEX = {}
for _i, _t in enumerate(_ALL_TRIPLES):
    _TRIPLE_INDEX[_t] = _i


def _rebuild_counts2(adj):
    """Faster rebuild: for each triple directly. n=20, 1140 triples -> cheap."""
    n = _N
    cnt = [0] * _NTRIPLES
    for idx, t in enumerate(_ALL_TRIPLES):
        # find the three vertices
        a = (t & -t).bit_length() - 1
        t2 = t ^ (1 << a)
        b = (t2 & -t2).bit_length() - 1
        t3 = t2 ^ (1 << b)
        c = t3.bit_length() - 1
        common = adj[a] & adj[b] & adj[c]
        common &= ~(1 << a) & ~(1 << b) & ~(1 << c)
        cnt[idx] = common.bit_count()
    return cnt


def _is_valid(adj):
    for c in _rebuild_counts2(adj):
        if c >= 3:
            return False
    return True


def _count_edges(adj):
    return sum(a.bit_count() for a in adj) // 2


def _try_add(adj, cnt, u, v):
    """Try to add edge (u,v). Returns True if valid (and updates cnt)."""
    if (adj[u] >> v) & 1:
        return True
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    # Only triples containing u or v are affected.
    # A triple {u, x, y} gains a common neighbor iff x,y are both neighbors of u
    #   AND v is a neighbor of x and y  -> v becomes a common neighbor.
    # A triple {u, x, v} gains a common neighbor iff x is neighbor of u and v.
    # General rule: adding edge (u,v) increases cnt for every triple {u,x,y}
    #   where x,y are common neighbors of u and v (v becomes common neighbor
    #   of u,x,y? No: v is adjacent to u by construction, and to x,y if x,y
    #   are neighbors of v). Let's just recompute affected triples cheaply.
    # Affected triples: any triple containing u or v.
    ok = True
    for idx, t in enumerate(_ALL_TRIPLES):
        if (t >> u) & 1 or (t >> v) & 1:
            a = (t & -t).bit_length() - 1
            t2 = t ^ (1 << a)
            b = (t2 & -t2).bit_length() - 1
            t3 = t2 ^ (1 << b)
            c = t3.bit_length() - 1
            common = adj[a] & adj[b] & adj[c]
            common &= ~(1 << a) & ~(1 << b) & ~(1 << c)
            nv = common.bit_count()
            if nv >= 3:
                ok = False
                break
            cnt[idx] = nv
    if not ok:
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return False
    return True


def _try_remove(adj, cnt, u, v):
    """Remove edge (u,v), always valid (removing edges can't create K3,3)."""
    if not ((adj[u] >> v) & 1):
        return False
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)
    for idx, t in enumerate(_ALL_TRIPLES):
        if (t >> u) & 1 or (t >> v) & 1:
            a = (t & -t).bit_length() - 1
            t2 = t ^ (1 << a)
            b = (t2 & -t2).bit_length() - 1
            t3 = t2 ^ (1 << b)
            c = t3.bit_length() - 1
            common = adj[a] & adj[b] & adj[c]
            common &= ~(1 << a) & ~(1 << b) & ~(1 << c)
            cnt[idx] = common.bit_count()
    return True


def _greedy_add(adj, cnt, deadline, rng):
    """Greedily add any missing edge that keeps the graph valid."""
    added = True
    while added and time.time() < deadline:
        added = False
        order = _ALL_PAIRS[:]
        rng.shuffle(order)
        for u, v in order:
            if time.time() > deadline:
                return
            if not ((adj[u] >> v) & 1):
                if _try_add(adj, cnt, u, v):
                    added = True


def _local_search(adj, cnt, deadline, rng):
    """Local search with 1-for-1 removal and 2-for-2 swaps to escape optima."""
    n = _N
    best = _count_edges(adj)
    while time.time() < deadline:
        # Phase A: try to add any missing edge
        _greedy_add(adj, cnt, deadline, rng)
        cur = _count_edges(adj)
        if cur > best:
            best = cur

        if time.time() > deadline:
            break

        # Phase B: try 2-for-2 swaps: remove (a,b) and (c,d), add (a,c),(b,d)
        # or other re-wiring. Try a few random swap attempts.
        existing = [(i, j) for i in range(n) for j in range(i + 1, n)
                    if (adj[i] >> j) & 1]
        missing = [(i, j) for i in range(n) for j in range(i + 1, n)
                   if not ((adj[i] >> j) & 1)]
        if not existing or not missing:
            break

        improved = False
        for _ in range(200):
            if time.time() > deadline:
                break
            # pick two existing edges to remove
            if len(existing) < 2:
                break
            (a, b), (c, d) = rng.sample(existing, 2)
            # candidate new edges: cross-wire
            candidates = []
            for x, y in [(a, c), (a, d), (b, c), (b, d)]:
                if x != y and (x, y) not in [(a, b), (c, d)]:
                    lo, hi = (x, y) if x < y else (y, x)
                    if not ((adj[lo] >> hi) & 1):
                        candidates.append((lo, hi))
            if len(candidates) < 2:
                continue
            # try removing both, adding two distinct candidates
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    e1 = candidates[i]
                    e2 = candidates[j]
                    # ensure no overlap with removed edges
                    if (e1 == (a, b)) or (e1 == (c, d)) or (e2 == (a, b)) or (e2 == (c, d)):
                        continue
                    if e1[0] == e2[0] and e1[1] == e2[1]:
                        continue
                    # save state
                    adj_save = adj[:]
                    cnt_save = cnt[:]
                    _try_remove(adj, cnt, a, b)
                    _try_remove(adj, cnt, c, d)
                    if _try_add(adj, cnt, e1[0], e1[1]) and _try_add(adj, cnt, e2[0], e2[1]):
                        newc = _count_edges(adj)
                        if newc > best:
                            best = newc
                            improved = True
                            # rebuild existing/missing lists
                            existing = [(i, j) for i in range(n) for j in range(i + 1, n)
                                        if (adj[i] >> j) & 1]
                            missing = [(i, j) for i in range(n) for j in range(i + 1, n)
                                       if not ((adj[i] >> j) & 1)]
                            break
                        else:
                            adj[:] = adj_save
                            cnt[:] = cnt_save
                    else:
                        adj[:] = adj_save
                        cnt[:] = cnt_save
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            # Phase C: random removal to escape local optimum, then re-greedy
            if existing:
                u, v = rng.choice(existing)
                _try_remove(adj, cnt, u, v)
                _greedy_add(adj, cnt, deadline, rng)
                cur = _count_edges(adj)
                if cur < best:
                    # didn't help; we keep the best state? We lost it.
                    # Re-run greedy to recover as much as possible.
                    pass
                best = max(best, _count_edges(adj))


def construct():
    n = _N
    start = time.time()
    deadline = start + 110.0  # safety margin under 120s (and 150s kill)

    best_adj = None
    best_count = 0

    # --- Build a strong seed: K_{2,18} plus a few safe extra edges ---
    # K2,18: partition {0,1} and the other 18. All cross edges present.
    # This is K3,3-free (needs 3 on each side to form K3,3).
    seed_adj = [0] * n
    for a in (0, 1):
        for v in range(2, n):
            seed_adj[a] |= (1 << v)
            seed_adj[v] |= (1 << a)
    # Add edge (0,1) safely? It creates a triangle but K3,3 needs bipartite.
    # (0,1) plus the 18 common neighbors: triple {0,1,x} has common neighbors
    # = the other 17 of the 18 -> >=3 => K3,3! So do NOT add (0,1).
    # Instead add edges among the 18 that are safe.
    seed_cnt = _rebuild_counts2(seed_adj)

    # Greedily fill the seed
    rng = random.Random(12345)
    _greedy_add(seed_adj, seed_cnt, deadline, rng)

    if _is_valid(seed_adj):
        c = _count_edges(seed_adj)
        if c > best_count:
            best_count = c
            best_adj = seed_adj[:]

    # --- Multiple restarts: random sparse starts + greedy + local search ---
    n_restarts = 24
    per = (deadline - time.time()) / max(1, n_restarts)
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        rng = random.Random(1000 + r * 7919)
        t0 = time.time()
        local_deadline = min(deadline, t0 + per)

        # random start, varying density
        p = 0.15 + 0.1 * (r % 4)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        # clean if invalid
        if not _is_valid(adj):
            cnt = _rebuild_counts2(adj)
            # remove random edges until valid
            for _ in range(400):
                if _is_valid(adj):
                    break
                existing = [(i, j) for i in range(n) for j in range(i + 1, n)
                            if (adj[i] >> j) & 1]
                if not existing:
                    break
                u, v = rng.choice(existing)
                _try_remove(adj, cnt, u, v)
        if not _is_valid(adj):
            continue

        cnt = _rebuild_counts2(adj)
        _greedy_add(adj, cnt, local_deadline, rng)
        _local_search(adj, cnt, local_deadline, rng)

        if _is_valid(adj):
            c = _count_edges(adj)
            if c > best_count:
                best_count = c
                best_adj = adj[:]

    # Final safety: if we still have time, polish the best with local search
    if best_adj is not None and time.time() < deadline:
        cnt = _rebuild_counts2(best_adj)
        rng = random.Random(99999)
        _local_search(best_adj, cnt, deadline, rng)
        if _is_valid(best_adj):
            best_count = _count_edges(best_adj)

    if best_adj is None:
        # Fallback: star K_{1,19}
        best_adj = [0] * n
        for i in range(1, n):
            best_adj[0] |= (1 << i)
            best_adj[i] |= (1 << 0)

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    return edges


# EVOLVE-BLOCK-END
