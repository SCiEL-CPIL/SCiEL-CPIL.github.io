"""Search-based solver for QAPLIB wil100: fast vectorized 2-opt/3-opt local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Exact integer cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _row_contribs(a, b, p):
    """Return row_contribs[i] = sum_j a[i,j] * b[p[i], p[j]] for each facility i."""
    bpi = b[p]  # bpi[i, j] = b[p[i], p[j]]
    return np.einsum('ij,ik->i', a, bpi, optimize=True)


def _swap_delta(a, b, p, i, j, row_contribs):
    """
    Compute cost delta when swapping facilities i and j in the permutation.
    After swap: p[i]<->p[j], i.e., facility i goes to loc p[j] and facility j goes to loc p[i].
    
    Cost = sum_k,l a[k,l] * b[p[k], p[l]]
    Only terms involving facility i or j change (as row index) or as column index.
    """
    li = p[i]
    lj = p[j]
    if li == lj:
        return 0
    
    # For each facility k, the change in its contribution to cost when we swap locations of i and j:
    # Before: ... a[k,i]*b[p[k],li] + a[k,j]*b[p[k],lj] + a[i,k]*b[li,p[k]] + a[j,k]*b[lj,p[k]] + a[i,j]*b[li,lj] + a[j,i]*b[lj,li]
    # After:  ... a[k,i]*b[p[k],lj] + a[k,j]*b[p[k],li] + a[i,k]*b[lj,p[k]] + a[j,k]*b[li,p[k]] + a[i,j]*b[lj,li] + a[j,i]*b[li,lj]
    
    # Let's compute the delta more carefully.
    # Delta = (new - old)
    
    # Terms where k is neither i nor j:
    # For row k: a[k,i]*(b[p[k],lj] - b[p[k],li]) + a[k,j]*(b[p[k],li] - b[p[k],lj])
    #          = (a[k,j] - a[k,i]) * (b[p[k],li] - b[p[k],lj])
    # For col k (i.e., terms a[i,k] and a[j,k]): a[i,k]*(b[lj,p[k]] - b[li,p[k]]) + a[j,k]*(b[li,p[k]] - b[lj,p[k]])
    #          = (a[j,k] - a[i,k]) * (b[li,p[k]] - b[lj,p[k]])
    #          = (a[j,k] - a[i,k]) * (b[p[i],p[k]] - b[p[j],p[k]])  ... wait, b is symmetric? Not necessarily.
    
    # Actually let me just compute it directly for correctness, using the fact that only i and j's positions change.
    
    # The total cost is sum_{k,l} a[k,l] * b[p[k], p[l]]
    # When we swap p[i] and p[j], the only terms that change are those where k=i or k=j or l=i or l=j.
    
    # Let's use a vectorized approach:
    # delta = sum over all pairs involving i or j of (new - old)
    
    # Terms with row i:
    # Old: sum_l a[i,l] * b[li, p[l]]
    # New: sum_l a[i,l] * b[lj, p[l]]
    # Delta_i_row = sum_l a[i,l] * (b[lj, p[l]] - b[li, p[l]])
    
    # Terms with row j:
    # Old: sum_l a[j,l] * b[lj, p[l]]
    # New: sum_l a[j,l] * b[li, p[l]]
    # Delta_j_row = sum_l a[j,l] * (b[li, p[l]] - b[lj, p[l]])
    
    # Terms with col i (l=i), excluding row i (already counted):
    # Old: sum_{k≠i} a[k,i] * b[p[k], li]
    # New: sum_{k≠i} a[k,i] * b[p[k], lj]
    # Delta_col_i = sum_{k≠i} a[k,i] * (b[p[k], lj] - b[p[k], li])
    
    # Terms with col j (l=j), excluding row j:
    # Old: sum_{k≠j} a[k,j] * b[p[k], lj]
    # New: sum_{k≠j} a[k,j] * b[p[k], li]
    # Delta_col_j = sum_{k≠j} a[k,j] * (b[p[k], li] - b[p[k], lj])
    
    # But wait, I'm double counting. Let me think again.
    
    # Actually, the cleanest way:
    # The cost change when swapping positions of i and j:
    # All terms where at least one index is i or j change.
    
    # Let me just compute it as:
    # delta = sum_l a[i,l]*(b[lj,p[l]] - b[li,p[l]]) + sum_l a[j,l]*(b[li,p[l]] - b[lj,p[l]])
    #       + sum_{k≠i,j} a[k,i]*(b[p[k],lj] - b[p[k],li]) + sum_{k≠i,j} a[k,j]*(b[p[k],li] - b[p[k],lj])
    #       + a[i,j]*(b[lj,li] - b[li,lj]) + a[j,i]*(b[li,lj] - b[lj,li])
    
    # Simplifying:
    # = sum_l a[i,l]*(b[lj,p[l]] - b[li,p[l]]) + sum_l a[j,l]*(b[li,p[l]] - b[lj,p[l]])
    #   + sum_{k≠i,j} (a[k,i] - a[k,j])*(b[p[k],lj] - b[p[k],li])
    #   + (a[i,j] - a[j,i])*(b[lj,li] - b[li,lj])
    
    # Note that for l=i in the first sum: a[i,i]*(b[lj,li] - b[li,li])... wait no.
    # When l=i in sum_l a[i,l]*(b[lj,p[l]] - b[li,p[l]]): p[l]=p[i]=li, so b[lj,li] - b[li,li]
    # When l=j in first sum: a[i,j]*(b[lj,p[j]] - b[li,p[j]]) = a[i,j]*(b[lj,lj] - b[li,lj])
    
    # This is getting complex. Let me just use a simpler but still efficient approach:
    # Compute the delta by summing over all k,l where the term changes.
    
    # Actually, the most efficient correct approach for n=100:
    # delta = (sum over all pairs involving i or j of new - old)
    
    # Let me just do it the straightforward vectorized way:
    pi = p.copy()
    pi[i], pi[j] = pi[j], pi[i]
    
    # Only rows i and j and columns i and j change
    # Actually all terms a[k,l]*b[p[k],p[l]] where p[k] or p[l] changed, i.e., k=i or k=j or l=i or l=j
    
    # The set of (k,l) pairs that change: {(k,l) : k in {i,j} or l in {i,j}}
    # That's 2n - 1 pairs.
    
    # For each such pair, compute old - new contribution.
    # old_contrib = a[k,l] * b[p[k], p[l]]
    # new_contrib = a[k,l] * b[pi[k], pi[l]]
    
    # We can compute this as:
    # delta = sum_{k in {i,j} or l in {i,j}} a[k,l] * (b[pi[k], pi[l]] - b[p[k], p[l]])
    
    # To avoid double counting, note that the set is {i,j} x {i,j} union {i,j} x (rest) union (rest) x {i,j}
    # = ({i,j} x all) union (all x {i,j}) minus ({i,j} x {i,j}) to avoid double count
    
    # Simpler: just iterate over the relevant indices.
    # Since n=100, we can afford a small loop.
    
    idx = np.array([i, j], dtype=np.int64)
    # All pairs where at least one is in idx
    ks = np.concatenate([idx, np.delete(np.arange(n), idx)])
    # For each k in ks, l ranges over all, but we only need l in idx or k in idx
    # Let me just compute for k in {i,j}: all l, and for l in {i,j}: k not in {i,j}
    
    total_delta = 0.0
    for k in [i, j]:
        for l in range(n):
            old_val = a[k, l] * b[p[k], p[l]]
            new_val = a[k, l] * b[pi[k], pi[l]]
            total_delta += (new_val - old_val)
    for k in range(n):
        if k == i or k == j:
            continue
        for l in [i, j]:
            old_val = a[k, l] * b[p[k], p[l]]
            new_val = a[k, l] * b[pi[k], pi[l]]
            total_delta += (new_val - old_val)
    
    return total_delta


def _local_search_2opt(a, b, p, row_contribs, time_budget, rng):
    """
    Fast 2-opt local search. Uses the swap delta computation.
    Returns (p, row_contribs, cost).
    """
    n = len(p)
    p = p.copy()
    row_contribs = row_contribs.copy()
    cost = _qap_cost(a, b, p)
    start = time.time()
    improved = True
    
    while improved and (time.time() - start) < time_budget:
        improved = False
        # Try all pairs in random order
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        rng.shuffle(pairs)
        
        for i, j in pairs:
            if (time.time() - start) > time_budget:
                break
            delta = _swap_delta(a, b, p, i, j, row_contribs)
            if delta < -1e-9:
                # Accept the swap
                p[i], p[j] = p[j], p[i]
                cost += delta
                # Update row_contribs: rows i and j change
                bpi = b[p]
                row_contribs[i] = np.dot(a[i], bpi[i])
                row_contribs[j] = np.dot(a[j], bpi[j])
                improved = True
    
    return p, row_contribs, cost


def _3opt_move(a, b, p, i, j, k):
    """
    Try cyclic shift: move facility at position i to position k, shifting j..k-1 down.
    Or more generally, reverse the segment [i..k].
    Let's do reversal of segment [i..k].
    """
    pi = p.copy()
    pi[i:k+1] = pi[i:k+1][::-1]
    # Compute cost difference
    # Only positions i..k change
    delta = 0
    for pos in range(i, k+1):
        for other in range(n):
            if other < i or other > k:
                # p[other] unchanged, but b[p[pos], p[other]] might change if pos in [i,k]
                # and b[p[other], p[pos]] might change
                delta += a[pos, other] * (b[pi[pos], pi[other]] - b[p[pos], p[other]])
                delta += a[other, pos] * (b[pi[other], pi[pos]] - b[p[other], p[pos]])
    # This is O(n * segment_length), could be slow. Let's just compute full cost difference.
    # Actually for n=100, let's just compute the delta properly.
    # The pairs that change: those where at least one position is in [i,k].
    
    # Let me just compute old and new for the affected rows/cols.
    # Positions affected: i..k
    affected = set(range(i, k+1))
    old_cost = 0
    new_cost = 0
    for pos in range(n):
        for other in range(n):
            if pos in affected or other in affected:
                old_cost += a[pos, other] * b[p[pos], p[other]]
                new_cost += a[pos, other] * b[pi[pos], pi[other]]
    delta = new_cost - old_cost
    return delta


def _local_search_2opt_fast(a, b, p, time_budget, rng):
    """
    Optimized 2-opt local search with precomputed structures.
    """
    n = len(p)
    p = p.copy()
    cost = _qap_cost(a, b, p)
    start = time.time()
    improved = True
    
    # Precompute b rows for fast access
    while improved and (time.time() - start) < time_budget:
        improved = False
        order = np.arange(n)
        rng.shuffle(order)
        
        for i in order:
            if (time.time() - start) > time_budget:
                break
            for j in order:
                if i >= j:
                    continue
                if (time.time() - start) > time_budget:
                    break
                
                delta = _swap_delta(a, b, p, i, j, None)
                if delta < -1e-9:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
    
    return p, cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    rng = np.random.RandomState(42)
    
    best_p = None
    best_cost = np.inf
    
    # Generate initial solutions
    initial_solutions = []
    
    # 1. Rank-matching: heavy flows to central locations
    heavy = np.argsort(-a.sum(axis=1), kind='stable')
    central = np.argsort(b.sum(axis=1), kind='stable')
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    initial_solutions.append(p1.copy())
    
    # 2. Reverse rank-matching
    p2 = np.empty(n, dtype=np.int64)
    p2[heavy] = central[::-1]
    initial_solutions.append(p2.copy())
    
    # 3-7. Random initial solutions
    for _ in range(5):
        pr = np.arange(n, dtype=np.int64)
        rng.shuffle(pr)
        initial_solutions.append(pr.copy())
    
    # 8. Greedy construction
    def _greedy_construct(a, b, rng):
        n = len(a)
        p = np.empty(n, dtype=np.int64)
        used = np.zeros(n, dtype=bool)
        
        for fac in range(n):
            # Choose the location that minimizes the incremental cost
            best_loc = -1
            best_cost = np.inf
            for loc in range(n):
                if used[loc]:
                    continue
                # Cost of placing fac at loc, considering already placed facilities
                inc_cost = 0
                for other in range(fac):
                    p_other = p[other]
                    inc_cost += a[fac, other] * b[loc, p_other]
                    inc_cost += a[other, fac] * b[p_other, loc]
                if inc_cost < best_cost:
                    best_cost = inc_cost
                    best_loc = loc
            p[fac] = best_loc
            used[best_loc] = True
        return p
    
    if time.time() - start_time < deadline - 10:
        p8 = _greedy_construct(a, b, rng)
        initial_solutions.append(p8)
    
    # Local search on each initial solution
    for p_init in initial_solutions:
        if time.time() - start_time > deadline - 15:
            break
        
        cost_init = _qap_cost(a, b, p_init)
        if cost_init < best_cost:
            best_cost = cost_init
            best_p = p_init.copy()
        
        remaining = max(1.0, deadline - 20 - (time.time() - start_time))
        p_ls, cost_ls = _local_search_2opt_fast(a, b, p_init.copy(), remaining, rng)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
