"""Biclique partition: Enhanced Greedy Set-Cover with Pair-Seeds, Restarts, and Polish."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

_N = P['n']
_ADJ = [set() for _ in range(_N)]
_EDGE_SET = set()
for _a, _b in P['edges']:
    _ADJ[_a].add(_b)
    _ADJ[_b].add(_a)
    _EDGE_SET.add((min(_a, _b), max(_a, _b)))

def _star_partition():
    """Trivial valid partition: one biclique per vertex covering edges to higher-index neighbors."""
    res = []
    for u in range(_N):
        nbrs = [v for v in _ADJ[u] if v > u]
        if nbrs:
            res.append(([u], list(nbrs)))
    return res

def _valid_partition(bics):
    """Return True iff bics is an exact, non-overlapping, host-edge biclique partition."""
    covered = set()
    for L, R in bics:
        if not L or not R:
            return False
        if set(L) & set(R):
            return False
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                if e not in _EDGE_SET:
                    return False
                if e in covered:
                    return False
                covered.add(e)
    return covered == _EDGE_SET

def _greedy_construct(deadline, random_seed=None):
    """Grow-based greedy set cover with Star, Edge-Seed, and Pair-Seed candidates."""
    if random_seed is not None:
        random.seed(random_seed)
    
    remaining = set(_EDGE_SET)
    bics = []
    
    while remaining and time.time() < deadline:
        best_L = None
        best_R = None
        best_count = 0
        best_rand = random.random() # For tie-breaking

        # 1. Single-vertex left sides (Stars)
        for u in range(_N):
            R = [v for v in _ADJ[u] if (min(u, v), max(u, v)) in remaining]
            if len(R) > best_count:
                best_count = len(R)
                best_L = [u]
                best_R = R
                best_rand = random.random()
            elif len(R) == best_count and best_L is not None:
                # Optional: random tie-breaking for diversity
                if random.random() > 0.5:
                    best_L = [u]
                    best_R = R
                    best_rand = random.random()

        # 2. Seed-edge bicliques: L = {a}, R = remaining-neighbors of a, then grow L.
        # We iterate over a subset of remaining edges to save time, or all if small.
        # For N=30, M~180, iterating all is fine.
        for e in list(remaining):
            if time.time() >= deadline:
                break
            a, b = e
            # R starts as neighbors of 'a' that have remaining edges
            R = [v for v in _ADJ[a] if (min(a, v), max(a, v)) in remaining]
            if not R:
                continue
            Rset = set(R)
            L = [a]
            Lset = set(L)
            # Grow L by adding vertices that connect to every vertex in R (and whose edges to R remain).
            changed = True
            while changed:
                changed = False
                for u in _ADJ[b]: # Look at neighbors of b as candidates for L
                    if u in Lset:
                        continue
                    # Check if u connects to all in R with remaining edges
                    valid = True
                    for v in R:
                        if (min(u, v), max(u, v)) not in remaining:
                            valid = False
                            break
                    if valid:
                        L.append(u)
                        Lset.add(u)
                        changed = True
            
            count = len(L) * len(R)
            if count > best_count:
                best_count = count
                best_L = L
                best_R = R
                best_rand = random.random()
            elif count == best_count and best_L is not None:
                if random.random() > 0.5:
                    best_L = L
                    best_R = R
                    best_rand = random.random()

        # 3. Pair-seeded bicliques: L = {u, v}, R = common remaining neighbors.
        # This helps find dense K_{2,k} structures.
        # To avoid O(M^2) explosion, we only try pairs that are edges in remaining.
        for e in list(remaining):
            if time.time() >= deadline:
                break
            u, v = e
            # R = neighbors of u in remaining AND neighbors of v in remaining
            R_u = [w for w in _ADJ[u] if (min(u, w), max(u, w)) in remaining]
            R_v = [w for w in _ADJ[v] if (min(v, w), max(v, w)) in remaining]
            
            # Intersection
            R = [w for w in R_u if w in R_v and w != u and w != v]
            
            L = [u, v]
            Lset = set(L)
            
            # Try to grow L further?
            changed = True
            while changed:
                changed = False
                # Candidates for L: neighbors of any node in L? Or just scan all?
                # Scan all is O(N * |R|). N=30, |R|<=30. 900 ops. Fast.
                for w in range(_N):
                    if w in Lset:
                        continue
                    valid = True
                    for r in R:
                        if (min(w, r), max(w, r)) not in remaining:
                            valid = False
                            break
                    if valid:
                        L.append(w)
                        Lset.add(w)
                        changed = True
            
            count = len(L) * len(R)
            if count > best_count:
                best_count = count
                best_L = L
                best_R = R
                best_rand = random.random()
            elif count == best_count and best_L is not None:
                if random.random() > 0.5:
                    best_L = L
                    best_R = R
                    best_rand = random.random()

        if best_L is None or best_count == 0:
            e = next(iter(remaining))
            best_L = [e[0]]
            best_R = [e[1]]
            best_count = 1

        bics.append((best_L, best_R))
        for u in best_L:
            for v in best_R:
                remaining.discard((min(u, v), max(u, v)))

    # Safety: any leftover edges become single-edge bicliques.
    for e in list(remaining):
        bics.append(([e[0]], [e[1]]))
    return bics

def _try_merge(bics, i, j):
    """Try to merge bicliques i and j into a single valid biclique. Return new list or None."""
    L1, R1 = bics[i]
    L2, R2 = bics[j]
    L = list(set(L1) | set(L2))
    R = list(set(R1) | set(R2))
    if not L or not R:
        return None
    if set(L) & set(R):
        return None
    for u in L:
        for v in R:
            if (min(u, v), max(u, v)) not in _EDGE_SET:
                return None
    # The merged biclique's edge set must equal the union of the two (no new edges).
    e1 = set()
    for u in L1:
        for v in R1:
            e1.add((min(u, v), max(u, v)))
    e2 = set()
    for u in L2:
        for v in R2:
            e2.add((min(u, v), max(u, v)))
    eu = set()
    for u in L:
        for v in R:
            eu.add((min(u, v), max(u, v)))
    if eu != (e1 | e2):
        return None
    new_bics = [b for k, b in enumerate(bics) if k not in (i, j)]
    new_bics.append((L, R))
    return new_bics

def _try_split(bics, i):
    """
    Attempt to split biclique i into two smaller bicliques that together cover the same edges.
    This is useful if the greedy step was too aggressive or if a better partition exists within the biclique.
    Returns (new_bics, True) or (None, False).
    """
    L, R = bics[i]
    edges = set()
    for u in L:
        for v in R:
            edges.add((min(u, v), max(u, v)))
            
    # Try all partitions of L into L1, L2
    # L is small, usually < 10. 2^|L| is manageable.
    # We only care about non-empty proper subsets.
    L_list = list(L)
    n_L = len(L_list)
    if n_L < 2:
        return None, False
        
    # Iterate over subsets of L
    # Optimization: only consider subsets that are "connected" or meaningful? 
    # For now, brute force subsets of size 1 to n_L-1.
    # To avoid duplicates (L1,L2) and (L2,L1), we only consider subsets containing the first element?
    # No, the split is not ordered, but the bicliques themselves are distinct objects.
    # However, if we find a split, we just replace i with two new bicliques.
    
    # Heuristic: Try moving one vertex from L to R? Or splitting L?
    # Let's try splitting L.
    for mask in range(1, (1 << n_L) - 1):
        # Skip symmetric masks? Not strictly necessary if we just check validity, but saves time.
        # Actually, if we find ANY valid split, we return it.
        L1 = []
        L2 = []
        for k in range(n_L):
            if mask & (1 << k):
                L1.append(L_list[k])
            else:
                L2.append(L_list[k])
        
        if not L1 or not L2:
            continue
            
        # Check if (L1, R) and (L2, R) are valid bicliques
        # Note: R is the same for both.
        # Check 1: All edges in (L1, R) must be in _EDGE_SET
        valid1 = True
        for u in L1:
            for v in R:
                if (min(u, v), max(u, v)) not in _EDGE_SET:
                    valid1 = False
                    break
            if not valid1:
                break
        if not valid1:
            continue
            
        # Check 2: All edges in (L2, R) must be in _EDGE_SET
        valid2 = True
        for u in L2:
            for v in R:
                if (min(u, v), max(u, v)) not in _EDGE_SET:
                    valid2 = False
                    break
            if not valid2:
                break
        if not valid2:
            continue
        
        # Check 3: The union of edges must exactly match the original edges.
        # Since L1 and L2 are disjoint and union to L, and R is the same,
        # the edge sets are disjoint and their union is the original edge set.
        # So this is automatically satisfied if valid1 and valid2 hold.
        # Wait, are the edge sets disjoint?
        # Edges in B1: {(u,v) | u in L1, v in R}
        # Edges in B2: {(u,v) | u in L2, v in R}
        # Since L1 and L2 are disjoint, no edge can be in both (unless u is in both, which it isn't).
        # So yes, disjoint.
        
        # Construct new bicliques
        new_bics = [b for k, b in enumerate(bics) if k != i]
        new_bics.append((L1, R))
        new_bics.append((L2, R))
        
        return new_bics, True
        
    return None, False

def _local_search(bics, deadline):
    """Repeatedly merge any pair of bicliques until no improvement or deadline."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(bics)
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                new_bics = _try_merge(bics, i, j)
                if new_bics is not None:
                    bics = new_bics
                    improved = True
                    break
            if improved:
                break
    return bics

def _polish(bics, deadline):
    """
    A local search step that tries to split bicliques to see if it allows subsequent merges
    or if the split itself is part of a better global structure.
    However, splitting increases the count. We only accept a split if it leads to a net improvement
    after subsequent merges.
    Given the time limit, we will try splits on the largest bicliques and see if merges follow.
    """
    current_bics = bics
    current_count = len(current_bics)
    
    # Sort bicliques by size (descending) to try splitting the biggest ones first
    indices = sorted(range(len(current_bics)), key=lambda i: len(current_bics[i][0]) * len(current_bics[i][1]), reverse=True)
    
    for idx in indices:
        if time.time() >= deadline:
            break
        
        new_bics, success = _try_split(current_bics, idx)
        if success:
            # Try to merge the resulting bicliques with others
            # We don't want to permanently increase the count unless we can merge back to a smaller count.
            # So, run a limited local search on the new_bics.
            polished = _local_search(new_bics, deadline)
            if len(polished) < current_count:
                current_bics = polished
                current_count = len(current_bics)
            # If len(polished) == current_count, we might have swapped one biclique for two others that are better positioned?
            # But the objective is strictly minimizing count. So we only keep if strictly better.
            
    return current_bics

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_bics = None
    best_count = float('inf')
    
    # Perform multiple restarts with different random seeds
    max_restarts = 20
    
    for r in range(max_restarts):
        if time.time() > deadline - 10.0: # Leave 10s for polish and final checks
            break
            
        seed = 42 + r
        bics = _greedy_construct(deadline - 5.0, seed)
        bics = _local_search(bics, deadline - 5.0)
        
        # Polish step: try to split and re-merge
        bics = _polish(bics, deadline - 2.0)
        
        if not _valid_partition(bics):
            continue
            
        count = len(bics)
        if count < best_count:
            best_count = count
            best_bics = bics
            
        # Early stop if we found a very good solution?
        if best_count <= 10:
             pass

    if best_bics is None:
        # Fallback to deterministic star partition
        best_bics = _star_partition()
    
    result = [[list(map(int, L)), list(map(int, R))] for L, R in best_bics]
    return result
# EVOLVE-BLOCK-END
