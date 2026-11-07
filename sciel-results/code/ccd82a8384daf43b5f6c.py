"""Cochromatic partition via branch-and-bound set cover over maximal homogeneous sets."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

import time
import sys

def _maximal_sets(adj, n):
    """Return list of maximal homogeneous sets (each a clique or independent set)."""
    sets = []
    # Bron-Kerbosch for maximal cliques
    def bronk(R, Pset, X):
        if not Pset and not X:
            sets.append(list(R))
        for v in list(Pset):
            bronk(R + [v], [u for u in Pset if adj[v][u]], [u for u in X if adj[v][u]])
            Pset.remove(v)
            X.append(v)
    bronk([], list(range(n)), [])
    # Maximal independent sets = maximal cliques of complement
    comp_adj = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            comp_adj[i][j] = (not adj[i][j]) and (i != j)
    def bronk2(R, Pset, X):
        if not Pset and not X:
            sets.append(list(R))
        for v in list(Pset):
            bronk2(R + [v], [u for u in Pset if comp_adj[v][u]], [u for u in X if comp_adj[v][u]])
            Pset.remove(v)
            X.append(v)
    bronk2([], list(range(n)), [])
    return sets

def _chromatic_number(adj, n):
    """Compute chromatic number via subset DP (n <= 20)."""
    full = (1 << n) - 1
    # Precompute independent-set indicator for all subsets
    indep = [False] * (1 << n)
    indep[0] = True
    for mask in range(1, 1 << n):
        v = (mask & -mask).bit_length() - 1
        rest = mask ^ (1 << v)
        indep[mask] = indep[rest] and (adj[v] & rest) == 0
    # DP: dp[mask] = min chromatic number of subgraph induced by mask
    dp = [0] * (1 << n)
    for mask in range(1, 1 << n):
        if indep[mask]:
            dp[mask] = 1
            continue
        best = n
        sub = mask
        while sub:
            if indep[sub]:
                r = dp[mask ^ sub]
                if r + 1 < best:
                    best = r + 1
            sub = (sub - 1) & mask
        dp[mask] = best
    return dp[full]

def construct():
    n = P['n']
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin before 90s kill

    # Build adjacency bitmasks
    adj = [0]*n
    for a, b in P['edges']:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    # Build boolean adjacency for maximal set enumeration
    adj_bool = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                adj_bool[i][j] = bool(adj[i] >> j & 1)

    # Find all maximal homogeneous sets
    hom_sets = _maximal_sets(adj_bool, n)
    # Convert to bitmasks and deduplicate
    set_masks = []
    seen = set()
    for s in hom_sets:
        m = 0
        for v in s:
            m |= (1 << v)
        if m not in seen:
            seen.add(m)
            set_masks.append(m)

    full = (1 << n) - 1

    # Precompute for each vertex, which maximal sets contain it
    vertex_sets = [[] for _ in range(n)]
    for i, m in enumerate(set_masks):
        for v in range(n):
            if (m >> v) & 1:
                vertex_sets[v].append(i)

    # Greedy initial upper bound: first-fit
    es = {tuple(e) for e in P['edges']}
    parts = []
    for v in range(n):
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es or (w, u) in es for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    ub = len(parts)
    best_parts = [list(p) for p in parts]

    # Lower bound: chromatic number of the full graph and its complement
    lb_chromatic = _chromatic_number(adj, n)
    comp_adj = [0]*n
    for i in range(n):
        for j in range(n):
            if i != j and not adj_bool[i][j]:
                comp_adj[i] |= (1 << j)
    lb_cochromatic = max(lb_chromatic, _chromatic_number(comp_adj, n))

    sys.setrecursionlimit(10000)

    # Sort set masks by size descending
    set_masks_sorted = sorted(set_masks, key=lambda m: bin(m).count('1'), reverse=True)
    # Precompute: for each vertex, indices in set_masks_sorted
    vertex_sets_sorted = [[] for _ in range(n)]
    for i, m in enumerate(set_masks_sorted):
        for v in range(n):
            if (m >> v) & 1:
                vertex_sets_sorted[v].append(i)

    # Precompute popcount cache
    popcount_cache = {}
    def popcount(x):
        if x not in popcount_cache:
            popcount_cache[x] = bin(x).count('1')
        return popcount_cache[x]

    # Lower bound function for a remaining mask
    def lower_bound(remaining):
        if not remaining:
            return 0
        max_size = 0
        for m in set_masks_sorted:
            if m & ~remaining == 0:
                s = popcount(m)
                if s > max_size:
                    max_size = s
        if max_size == 0:
            return 1
        import math
        lb1 = math.ceil(popcount(remaining) / max_size)
        # Chromatic number of induced subgraph on remaining (if small enough)
        if popcount(remaining) <= 12:
            verts = [v for v in range(n) if (remaining >> v) & 1]
            k = len(verts)
            sub_adj = [0]*k
            for i, vi in enumerate(verts):
                for j, vj in enumerate(verts):
                    if i != j and (adj[vi] >> vj) & 1:
                        sub_adj[i] |= (1 << j)
            sub_chi = _chromatic_number(sub_adj, k)
            sub_comp = [0]*k
            for i in range(k):
                for j in range(k):
                    if i != j and not (sub_adj[i] >> j) & 1:
                        sub_comp[i] |= (1 << j)
            sub_chi_comp = _chromatic_number(sub_comp, k)
            lb2 = max(sub_chi, sub_chi_comp)
            return max(lb1, lb2)
        return lb1

    def bnb(remaining, depth, chosen_indices):
        nonlocal ub, best_parts
        if time.time() > deadline:
            return
        if not remaining:
            if depth < ub:
                ub = depth
                assign = [-1]*n
                for idx in chosen_indices:
                    m = set_masks_sorted[idx]
                    for v in range(n):
                        if (m >> v) & 1 and assign[v] == -1:
                            assign[v] = idx
                parts_new = [[] for _ in range(len(chosen_indices))]
                for v in range(n):
                    if assign[v] >= 0:
                        parts_new[assign[v]].append(v)
                parts_new = [p for p in parts_new if p]
                if len(parts_new) <= depth:
                    best_parts = parts_new
            return
        # Lower bound check
        lb = lower_bound(remaining)
        if depth + lb >= ub:
            return
        # MRV: pick vertex in remaining with fewest candidate sets
        best_v = -1
        best_cnt = n + 1
        for v in range(n):
            if (remaining >> v) & 1:
                cnt = 0
                for si in vertex_sets_sorted[v]:
                    m = set_masks_sorted[si]
                    if m & ~remaining == 0:
                        cnt += 1
                if cnt < best_cnt:
                    best_cnt = cnt
                    best_v = v
                if cnt == 1:
                    break
        if best_v == -1 or best_cnt == 0:
            return
        # Branch on best_v
        for si in vertex_sets_sorted[best_v]:
            if time.time() > deadline:
                return
            m = set_masks_sorted[si]
            if m & ~remaining:
                continue
            new_remaining = remaining & ~m
            chosen_indices.append(si)
            bnb(new_remaining, depth + 1, chosen_indices)
            chosen_indices.pop()
            if depth + 1 >= ub:
                return

    bnb(full, 0, [])

    # Ensure best_parts is a valid partition
    if not best_parts:
        return [list(p) for p in parts]
    seen = set()
    for part in best_parts:
        for v in part:
            if v in seen:
                return [list(p) for p in parts]
            seen.add(v)
    if seen != set(range(n)):
        return [list(p) for p in parts]
    for part in best_parts:
        if len(part) == 1:
            continue
        hits = []
        for i in range(len(part)):
            for j in range(i+1, len(part)):
                hits.append((part[i], part[j]) in es or (part[j], part[i]) in es)
        if not (all(hits) or not any(hits)):
            return [list(p) for p in parts]
    return [[int(v) for v in part] for part in best_parts]
# EVOLVE-BLOCK-END
