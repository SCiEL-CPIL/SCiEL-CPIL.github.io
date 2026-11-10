"""Cochromatic partition solver with branch-and-bound."""
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

N = P['n']
# adjacency bitmasks
adj = [0] * N
for a, b in P['edges']:
    adj[a] |= (1 << b)
    adj[b] |= (1 << a)
full_mask = (1 << N) - 1

# precompute max clique size for every subset (for lower bound)
# max_clique[S] = size of largest clique contained in S (in G)
max_clique = [0] * (1 << N)
# max_indep[S] = size of largest independent set contained in S (in G) = max clique in complement
# build complement adjacency
cadj = [0] * N
for v in range(N):
    cadj[v] = full_mask ^ (1 << v) ^ adj[v]

def _max_clique_mask(adjlist):
    # max clique size for all subsets via DP using one vertex removal
    mc = [0] * (1 << N)
    for S in range(1, 1 << N):
        # pick lowest set bit
        lb = S & (-S)
        v = lb.bit_length() - 1
        rest = S ^ lb
        # either exclude v, or include v (then only neighbors of v in rest)
        a = mc[rest]
        b = 1 + mc[rest & adjlist[v]]
        mc[S] = a if a > b else b
    return mc

max_clique = _max_clique_mask(adj)
max_indep = _max_clique_mask(cadj)  # independent set in G = clique in complement

def ceil_div(a, b):
    return (a + b - 1) // b

# ---------- seed feasible solution (greedy, like current) ----------
def _seed_parts():
    es = {tuple(e) for e in P['edges']}
    parts = []
    for v in range(N):
        placed = False
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i + 1:]]
            if all(hits) or not any(hits):
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return parts

# ---------- helpers ----------
def _is_clique(S):
    # S bitmask; true if induced subgraph is complete
    # every vertex in S adjacent to every other
    m = S
    while m:
        lb = m & (-m); v = lb.bit_length() - 1; m ^= lb
        if (adj[v] & S) != (S ^ (1 << v)):
            return False
    return True

def _is_independent(S):
    m = S
    while m:
        lb = m & (-m); v = lb.bit_length() - 1; m ^= lb
        if (adj[v] & S) != 0:
            return False
    return True

# group a set (bitmask) of same-type vertices into valid parts (cliques or ind-sets)
# returns list of parts (as lists of ints). Uses greedy maximal packing.
def _pack(mask, is_clique_type):
    # returns list of parts covering mask
    parts = []
    remaining = mask
    while remaining:
        # find a maximal valid part containing the lowest vertex of remaining
        lb = remaining & (-remaining)
        seed_v = lb.bit_length() - 1
        part_mask = lb
        # try to add vertices to make it maximal valid
        cand = remaining
        while True:
            added = False
            m = cand
            while m:
                b = m & (-m); u = b.bit_length() - 1; m ^= b
                if part_mask & b:
                    continue
                trial = part_mask | b
                ok = _is_clique(trial) if is_clique_type else _is_independent(trial)
                if ok:
                    part_mask = trial
                    added = True
            if not added:
                break
        parts.append(part_mask)
        remaining ^= part_mask
    return parts

def _mask_to_list(mask):
    out = []
    m = mask
    while m:
        lb = m & (-m); out.append(lb.bit_length() - 1); m ^= lb
    return out

# ---------- branch and bound ----------
def construct():
    t0 = time.time()
    DEADLINE = 75.0  # safety margin under 90s

    # initial feasible solution from seed
    seed = _seed_parts()
    best = len(seed)
    best_parts = [list(p) for p in seed]

    # order vertices for branching: pick a vertex that is "hard" (high degree helps)
    # we branch on unassigned vertices; use a fixed order by descending degree for determinism
    order = sorted(range(N), key=lambda v: -bin(adj[v]).count('1'))

    # state: label[v] in {0=unassigned, 1=clique, 2=independent}
    label = [0] * N
    # masks of assigned clique/independent
    Cmask = 0
    Imask = 0

    # precompute for lower bound: we need max_clique[Cmask | cand] etc.

    def lower_bound(Cm, Im):
        # number of clique parts needed for Cm vertices (clique type) = ceil(|Cm| / max_clique[Cm])
        # number of independent parts needed for Im = ceil(|Im| / max_indep[Im])
        cparts = ceil_div(Cm.bit_count(), max_clique[Cm]) if Cm else 0
        iparts = ceil_div(Im.bit_count(), max_indep[Im]) if Im else 0
        return cparts + iparts

    # recursion
    import sys
    sys.setrecursionlimit(10000)

    # We'll do iterative deepening on target K to allow early termination & clean bounds.
    # For a target K, search assignments where lower_bound <= K.
    # We'll try K from best-1 downward? Actually we want to find any solution with < best parts.
    # Use IDDFS: for K in range(1, best): if feasible(K) then best=K, break.

    # But computing parts from a full label assignment: pack Cmask into cliques, Imask into ind-sets, total parts = len(clique_parts)+len(ind_parts). Need <= K.

    def feasible(K):
        # returns (found, parts) if a valid partition with <=K parts exists
        nonlocal label, Cmask, Imask
        label = [0] * N
        Cmask = 0
        Imask = 0
        found_parts = None

        def rec(idx):
            nonlocal Cmask, Imask, found_parts
            if time.time() > DEADLINE:
                return False
            if found_parts is not None:
                return True
            if idx == N:
                # all assigned; compute parts count
                cparts = _pack(Cmask, True)
                iparts = _pack(Imask, False)
                total = len(cparts) + len(iparts)
                if total <= K:
                    # build final parts
                    parts = []
                    for pm in cparts:
                        parts.append(_mask_to_list(pm))
                    for pm in iparts:
                        parts.append(_mask_to_list(pm))
                    found_parts = parts
                    return True
                return False
            v = order[idx]
            # symmetry: first unassigned vertex must be clique
            if idx == 0:
                choices = [1]
            else:
                choices = [1, 2]
            for ch in choices:
                if ch == 1:
                    newC = Cmask | (1 << v)
                    lb = lower_bound(newC, Imask)
                    if lb > K:
                        continue
                    label[v] = 1
                    Cmask = newC
                    if rec(idx + 1):
                        return True
                    Cmask = Cmask ^ (1 << v)
                else:
                    newI = Imask | (1 << v)
                    lb = lower_bound(Cmask, newI)
                    if lb > K:
                        continue
                    label[v] = 2
                    Imask = newI
                    if rec(idx + 1):
                        return True
                    Imask = Imask ^ (1 << v)
                label[v] = 0
            return False

        rec(0)
        if found_parts is not None:
            return True, found_parts
        return False, None

    for K in range(1, best):
        if time.time() > DEADLINE:
            break
        found, parts = feasible(K)
        if found:
            best = K
            best_parts = parts
            break

    # convert to lists of ints, ensure plain types
    out = []
    for part in best_parts:
        out.append([int(x) for x in part])
    return out
