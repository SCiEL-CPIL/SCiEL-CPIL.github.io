"""QAP tho150 solver: spectral init + efficient tabu search with O(n) swap deltas."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_cost(a, b, p):
    """Exact cost sum_{i,j} a[i,j]*b[p[i],p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _precompute(a, b, p):
    """Precompute per-facility contributions for O(n) swap delta.
    For facility i at location p[i], its contribution to cost is:
      sum_j a[i,j]*b[p[i],p[j]] + sum_j a[j,i]*b[p[j],p[i]]
    = b[p[i], p].dot(a[:, i]) ... let's define:
      row_contrib[i] = sum_j a[i,j] * b[p[i], p[j]]  (outgoing from i)
      col_contrib[i] = sum_j a[j,i] * b[p[j], p[i]]  (incoming to i)
    total = sum_i (row_contrib[i] + col_contrib[i]) but this double counts? 
    Actually cost = sum_i sum_j a[i,j] b[p[i],p[j]]. For a swap of i and k,
    only terms involving i or k change.
    """
    # b_at_p = b[p[i], p[j]] for all i,j -> shape (n,n)
    b_at_p = b[np.ix_(p, p)]
    # row_contrib[i] = sum_j a[i,j] * b_at_p[i,j]
    row_contrib = np.einsum('ij,ij->i', a, b_at_p)
    # col_contrib[i] = sum_j a[j,i] * b_at_p[j,i]
    col_contrib = np.einsum('ji,ji->i', a, b_at_p)
    return row_contrib, col_contrib, b_at_p


def _swap_delta(i, k, a, b_at_p, row_contrib, col_contrib, p):
    """Compute change in cost if we swap facilities i and k.
    Cost = sum_{i,j} a[i,j] * b_at_p[i,j] where b_at_p[i,j] = b[p[i],p[j]].
    Swapping p[i] and p[k] changes b_at_p entries in row i, row k, col i, col k.
    New b_at_p':
      b_at_p'[i,j] = b[p[k], p[j]] for j != k
      b_at_p'[i,k] = b[p[k], p[i]]
      b_at_p'[k,j] = b[p[i], p[j]] for j != i
      b_at_p'[k,i] = b[p[i], p[k]]
      others unchanged.
    Delta = sum_j a[i,j]*(b_at_p'[i,j]-b_at_p[i,j]) + sum_j a[k,j]*(b_at_p'[k,j]-b_at_p[k,j])
          + sum_j a[j,i]*(b_at_p'[j,i]-b_at_p[j,i]) + sum_j a[j,k]*(b_at_p'[j,k]-b_at_p[j,k])
    But careful: terms (i,i),(i,k),(k,i),(k,k) appear in both row and col sums.
    Let's just compute the 4 affected rows/cols directly.
    """
    li = p[i]
    lk = p[k]
    # Affected: (i, all j), (k, all j), (all j, i), (all j, k)
    # But (i,i) appears in row i and col i; (i,k) in row i and col k; etc.
    # Easiest: compute old and new for the 4x4 affected sub-block plus the rest.
    # Actually, let's compute delta as:
    # For row i: sum_j a[i,j] * (b_at_p_new[i,j] - b_at_p[i,j])
    #   b_at_p_new[i,j] = b[lk, p[j]] if j != k, else b[lk, li]
    #   b_at_p[i,j] = b[li, p[j]]
    #   So for j != k: a[i,j]*(b[lk,p[j]] - b[li,p[j]])
    #   For j == k: a[i,k]*(b[lk,li] - b[li,lk])
    # Similarly row k, col i, col k, but avoid double counting.
    # Let me just recompute the affected part.
    # Affected indices in the (i,j) grid: {i,k} x {i,k} plus {i,k} x all and all x {i,k}
    # Simpler: delta = (new_cost - old_cost) restricted to affected entries.
    # Old affected = sum over (i,j) in {i,k}x{0..n-1} of a[i,j]*b_at_p[i,j]
    #              + sum over (j,i) in {0..n-1}x{i,k} of a[j,i]*b_at_p[j,i]
    #              - sum over (i,k) and (k,i) double counted... 
    # This is getting messy. Let me use a cleaner approach:
    # The cost can be written as sum_i (row_contrib[i] + col_contrib[i]) / 2 * 2? No.
    # cost = sum_i sum_j a[i,j] b_at_p[i,j]. This is NOT symmetric in general.
    # Let me just compute the delta by directly evaluating the changed terms.
    # Changed terms: all (r,c) where r in {i,k} OR c in {i,k}.
    # That's 2n - 1 terms (row i, row k, col i, col k, minus the 4 overlaps counted twice).
    # Let me just compute old and new for these.
    
    # Build the set of affected (r,c) pairs: row i, row k, col i, col k
    # old_val = sum of a[r,c]*b_at_p[r,c] for affected
    # new_val = sum of a[r,c]*b_at_p_new[r,c] for affected
    # b_at_p_new differs only where p changed: p[i]<->p[k]
    # b_at_p_new[r,c] = b[p_new[r], p_new[c]]
    # p_new = p with i,k swapped
    
    # For efficiency, compute:
    # delta_row_i = sum_j a[i,j]*(b[lk,p[j]] - b[li,p[j]])  [but p[j] for j=k becomes li]
    # Let's handle j=k separately.
    # For j != k: b[lk, p[j]] - b[li, p[j]]
    # For j == k: b[lk, li] - b[li, lk]
    
    # I'll compute using vectorized ops on the affected rows/cols.
    # Create p_new conceptually.
    # b_at_p_new[r,c] = b[p_new[r], p_new[c]]
    
    # Affected rows: i, k. Affected cols: i, k.
    # Let me compute the contribution of the full affected region.
    
    # Old affected contribution:
    # Row i: sum_j a[i,j] * b_at_p[i,j]
    # Row k: sum_j a[k,j] * b_at_p[k,j]
    # Col i (excluding row i,k already counted): sum_{j not in {i,k}} a[j,i] * b_at_p[j,i]
    # Col k (excluding row i,k already counted): sum_{j not in {i,k}} a[j,k] * b_at_p[j,k]
    
    # New affected contribution:
    # Row i: sum_j a[i,j] * b[p_new[i], p_new[j]] = sum_j a[i,j] * b[lk, p_new[j]]
    #   p_new[j] = p[j] if j not in {i,k}, else swapped.
    #   For j not in {i,k}: b[lk, p[j]]
    #   For j == i: b[lk, lk]  (since p_new[i] = lk)
    #   For j == k: b[lk, li]  (since p_new[k] = li)
    # Row k: sum_j a[k,j] * b[p_new[k], p_new[j]] = sum_j a[k,j] * b[li, p_new[j]]
    #   For j not in {i,k}: b[li, p[j]]
    #   For j == i: b[li, lk]
    #   For j == k: b[li, li]
    # Col i (j not in {i,k}): sum a[j,i] * b[p_new[j], p_new[i]] = sum a[j,i] * b[p[j], lk]
    # Col k (j not in {i,k}): sum a[j,k] * b[p_new[j], p_new[k]] = sum a[j,k] * b[p[j], li]
    
    # This is O(n) per swap. Let me implement it.
    n = len(p)
    # p_new for positions i and k
    # For row i:
    #   sum_{j not in {i,k}} a[i,j]*b[lk,p[j]] + a[i,i]*b[lk,lk] + a[i,k]*b[lk,li]
    # For row k:
    #   sum_{j not in {i,k}} a[k,j]*b[li,p[j]] + a[k,i]*b[li,lk] + a[k,k]*b[li,li]
    # For col i (j not in {i,k}): sum a[j,i]*b[p[j],lk]
    # For col k (j not in {i,k}): sum a[j,k]*b[p[j],li]
    
    # Old:
    #   sum_{j not in {i,k}} a[i,j]*b[li,p[j]] + a[i,i]*b[li,li] + a[i,k]*b[li,lk]
    #   + sum_{j not in {i,k}} a[k,j]*b[lk,p[j]] + a[k,i]*b[lk,li] + a[k,k]*b[lk,lk]
    #   + sum_{j not in {i,k}} a[j,i]*b[p[j],li]
    #   + sum_{j not in {i,k}} a[j,k]*b[p[j],lk]
    
    # Delta = new - old. Let me compute new and old separately using vectorized sums.
    # For "j not in {i,k}", I can sum over all j and subtract j=i and j=k.
    
    # Row i new: sum_j a[i,j]*b[lk,p_new[j]]
    #   = sum_j a[i,j]*b[lk,p[j]] - a[i,i]*b[lk,p[i]] - a[i,k]*b[lk,p[k]] + a[i,i]*b[lk,lk] + a[i,k]*b[lk,li]
    #   = sum_j a[i,j]*b[lk,p[j]] - a[i,i]*b[lk,li] - a[i,k]*b[lk,lk] + a[i,i]*b[lk,lk] + a[i,k]*b[lk,li]
    #   = sum_j a[i,j]*b[lk,p[j]] + a[i,i]*(b[lk,lk]-b[lk,li]) + a[i,k]*(b[lk,li]-b[lk,lk])
    # Row i old: sum_j a[i,j]*b[li,p[j]]
    #   (this is just the original row contribution, no swap)
    
    # This is getting complicated. Let me just directly compute using the b matrix lookups.
    
    # For speed, precompute nothing extra; just do the sums.
    # Let me define helper: for a given location loc, b[loc, p] is a vector.
    # b_lk_p = b[lk, p]  # shape (n,)
    # b_li_p = b[li, p]
    # b_p_lk = b[p, lk]  # shape (n,)
    # b_p_li = b[p, li]
    
    b_lk_p = b[lk, p]  # b[lk, p[j]] for all j
    b_li_p = b[li, p]
    b_p_lk = b[p, lk]  # b[p[j], lk] for all j
    b_p_li = b[p, li]
    
    # Row i new: sum_j a[i,j]*b[lk, p_new[j]]
    # p_new[j] = p[j] if j not in {i,k}; p_new[i]=lk, p_new[k]=li
    # = sum_{j not in {i,k}} a[i,j]*b[lk,p[j]] + a[i,i]*b[lk,lk] + a[i,k]*b[lk,li]
    # = (sum_j a[i,j]*b_lk_p[j]) - a[i,i]*b_lk_p[i] - a[i,k]*b_lk_p[k] + a[i,i]*b[lk,lk] + a[i,k]*b[lk,li]
    # b_lk_p[i] = b[lk, p[i]] = b[lk, li]
    # b_lk_p[k] = b[lk, p[k]] = b[lk, lk]
    row_i_new = np.dot(a[i, :], b_lk_p) - a[i,i]*b[lk,li] - a[i,k]*b[lk,lk] + a[i,i]*b[lk,lk] + a[i,k]*b[lk,li]
    # Row i old: sum_j a[i,j]*b[li,p[j]] = np.dot(a[i,:], b_li_p)
    row_i_old = np.dot(a[i, :], b_li_p)
    
    # Row k new: sum_j a[k,j]*b[li, p_new[j]]
    # = (sum_j a[k,j]*b_li_p[j]) - a[k,i]*b_li_p[i] - a[k,k]*b_li_p[k] + a[k,i]*b[li,lk] + a[k,k]*b[li,li]
    # b_li_p[i] = b[li, p[i]] = b[li, li]
    # b_li_p[k] = b[li, p[k]] = b[li, lk]
    row_k_new = np.dot(a[k, :], b_li_p) - a[k,i]*b[li,li] - a[k,k]*b[li,lk] + a[k,i]*b[li,lk] + a[k,k]*b[li,li]
    # Row k old: sum_j a[k,j]*b[lk,p[j]] = np.dot(a[k,:], b_lk_p)
    row_k_old = np.dot(a[k, :], b_lk_p)
    
    # Col i (j not in {i,k}) new: sum a[j,i]*b[p_new[j], lk]
    # p_new[j] = p[j] if j not in {i,k}
    # = sum_{j not in {i,k}} a[j,i]*b[p[j],lk] = (sum_j a[j,i]*b_p_lk[j]) - a[i,i]*b_p_lk[i] - a[k,i]*b_p_lk[k]
    # b_p_lk[i] = b[p[i], lk] = b[li, lk]
    # b_p_lk[k] = b[p[k], lk] = b[lk, lk]
    col_i_new = np.dot(a[:, i], b_p_lk) - a[i,i]*b[li,lk] - a[k,i]*b[lk,lk]
    # Col i old (j not in {i,k}): sum a[j,i]*b[p[j],li] = (sum_j a[j,i]*b_p_li[j]) - a[i,i]*b_p_li[i] - a[k,i]*b_p_li[k]
    # b_p_li[i] = b[p[i],li] = b[li,li]
    # b_p_li[k] = b[p[k],li] = b[lk,li]
    col_i_old = np.dot(a[:, i], b_p_li) - a[i,i]*b[li,li] - a[k,i]*b[lk,li]
    
    # Col k (j not in {i,k}) new: sum a[j,k]*b[p_new[j], li]
    # = sum_{j not in {i,k}} a[j,k]*b[p[j],li] = (sum_j a[j,k]*b_p_li[j]) - a[i,k]*b_p_li[i] - a[k,k]*b_p_li[k]
    # b_p_li[i] = b[li,li], b_p_li[k]=b[lk,li]
    col_k_new = np.dot(a[:, k], b_p_li) - a[i,k]*b[li,li] - a[k,k]*b[lk,li]
    # Col k old (j not in {i,k}): sum a[j,k]*b[p[j],lk] = (sum_j a[j,k]*b_p_lk[j]) - a[i,k]*b_p_lk[i] - a[k,k]*b_p_lk[k]
    # b_p_lk[i]=b[li,lk], b_p_lk[k]=b[lk,lk]
    col_k_old = np.dot(a[:, k], b_p_lk) - a[i,k]*b[li,lk] - a[k,k]*b[lk,lk]
    
    delta = (row_i_new + row_k_new + col_i_new + col_k_new) - (row_i_old + row_k_old + col_i_old + col_k_old)
    return int(delta)


def _spectral_init(a, b, n, rng):
    """Spectral heuristic: use eigenvectors of a to initialize assignment."""
    # Compute a few eigenvectors of a
    try:
        eigvals, eigvecs = np.linalg.eigh(a)
        # Use top few eigenvectors to create a continuous relaxation
        # Assign facilities based on the principal eigenvector
        v = eigvecs[:, -1]  # largest eigenvector
        # Rank facilities by v
        fac_order = np.argsort(-v, kind='stable')
        # For locations, use eigenvectors of b (smallest eigenvalue = most central)
        eigvals_b, eigvecs_b = np.linalg.eigh(b)
        vb = eigvecs_b[:, 0]  # smallest eigenvector
        loc_order = np.argsort(vb, kind='stable')
        p = np.empty(n, dtype=np.int64)
        p[fac_order] = loc_order
    except:
        # Fallback to rank matching
        heavy = np.argsort(-a.sum(axis=1), kind='stable')
        central = np.argsort(b.sum(axis=1), kind='stable')
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
    return p


def _tabu_search(a, b, p, max_iters, deadline, rng, tabu_len=20):
    """Tabu search with O(n) swap deltas."""
    n = len(p)
    p = p.copy()
    cost = _compute_cost(a, b, p)
    best_cost = cost
    best_p = p.copy()
    
    # Precompute
    row_contrib, col_contrib, b_at_p = _precompute(a, b, p)
    
    tabu_until = np.zeros(n, dtype=np.int64)  # tabu until iteration for each facility
    iter_count = 0
    
    while iter_count < max_iters:
        # Check deadline
        if time.time() > deadline:
            break
        
        # Find best swap
        best_delta = 0
        best_i, best_k = -1, -1
        best_aspiration = False
        
        # Sample a subset of swaps for speed (or all if n is small enough)
        # For n=150, all pairs = 150*149/2 = 11175 swaps, each O(n)=150 -> ~1.7M ops per iter
        # That's fast. Let's do all pairs.
        for i in range(n):
            if tabu_until[i] > iter_count and not (cost < best_cost):  # aspiration
                continue
            for k in range(i+1, n):
                if tabu_until[k] > iter_count and not (cost < best_cost):
                    continue
                delta = _swap_delta(i, k, a, b_at_p, row_contrib, col_contrib, p)
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_k = i, k
                    best_aspiration = (tabu_until[i] > iter_count or tabu_until[k] > iter_count)
        
        if best_i == -1:
            # No improving move found; try a random move to escape
            i, k = rng.choice(n, 2, replace=False)
            delta = _swap_delta(int(i), int(k), a, b_at_p, row_contrib, col_contrib, p)
            best_delta = delta
            best_i, best_k = int(i), int(k)
        
        # Perform swap
        i, k = best_i, best_k
        p[i], p[k] = p[k], p[i]
        cost += best_delta
        
        # Update tabu
        tabu_until[i] = iter_count + tabu_len
        tabu_until[k] = iter_count + tabu_len
        
        # Update precomputed
        row_contrib, col_contrib, b_at_p = _precompute(a, b, p)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
        
        iter_count += 1
    
    return best_p, best_cost


def solve(a, b):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    n = len(a)
    rng = np.random.default_rng(42)
    
    # Generate multiple initial solutions
    inits = []
    # Spectral
    inits.append(_spectral_init(a, b, n, rng))
    # Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind='stable')
    central = np.argsort(b.sum(axis=1), kind='stable')
    p_rm = np.empty(n, dtype=np.int64)
    p_rm[heavy] = central
    inits.append(p_rm)
    # Random
    inits.append(rng.permutation(n).astype(np.int64))
    
    best_p = inits[0].copy()
    best_cost = _compute_cost(a, b, best_p)
    
    # Evaluate all inits
    for p_init in inits:
        c = _compute_cost(a, b, p_init)
        if c < best_cost:
            best_cost = c
            best_p = p_init.copy()
    
    # Tabu search from best init
    if time.time() < deadline:
        p, c = _tabu_search(a, b, best_p, max_iters=5000, deadline=deadline, rng=rng)
        if c < best_cost:
            best_cost = c
            best_p = p.copy()
    
    # Try more restarts if time remains
    restart = 0
    while time.time() < deadline - 5:
        restart += 1
        # New random init
        p_init = rng.permutation(n).astype(np.int64)
        p, c = _tabu_search(a, b, p_init, max_iters=2000, deadline=deadline, rng=rng)
        if c < best_cost:
            best_cost = c
            best_p = p.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
