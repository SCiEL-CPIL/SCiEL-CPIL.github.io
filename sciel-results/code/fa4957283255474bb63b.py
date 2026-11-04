"""Rewritten QAP solver for tai256c with multi-start local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute the QAP objective sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    """
    Compute the change in QAP cost if we swap facilities at positions i and j
    in the permutation p. p maps facility -> location (p[fac] = loc).
    After swap: p'[i] = p[j], p'[j] = p[i], others unchanged.
    
    Cost = sum_{u,v} a[u,v] * b[p[u], p[v]]
    
    The terms that change are those involving u=i or v=i or u=j or v=j.
    Let's compute delta = cost_after - cost_before.
    
    Terms involving i or j (with multiplicity handled):
    For each (u,v) where u in {i,j} or v in {i,j}:
      before: a[u,v] * b[p[u], p[v]]
      after:  a[u,v] * b[p'[u], p'[v]]
    
    p'[i] = p[j], p'[j] = p[i]
    
    So we need to sum over all (u,v) with u in {i,j} or v in {i,j}:
      a[u,v] * (b[p'[u], p'[v]] - b[p[u], p[v]])
    
    Let li = p[i], lj = p[j].
    
    Cases:
    u=i, v=i: a[i,i]*(b[lj,lj] - b[li,li])
    u=i, v=j: a[i,j]*(b[lj,li] - b[li,lj])
    u=j, v=i: a[j,i]*(b[li,lj] - b[lj,li])
    u=j, v=j: a[j,j]*(b[li,li] - b[lj,lj])
    u=i, v=k (k!=i,j): a[i,k]*(b[lj,p[k]] - b[li,p[k]])
    u=k, v=i (k!=i,j): a[k,i]*(b[p[k],lj] - b[p[k],li])
    u=j, v=k (k!=i,j): a[j,k]*(b[li,p[k]] - b[lj,p[k]])
    u=k, v=j (k!=i,j): a[k,j]*(b[p[k],li] - b[p[k],lj])
    
    Combine:
    delta = a[i,i]*(b[lj,lj]-b[li,li]) + a[j,j]*(b[li,li]-b[lj,lj])
          + a[i,j]*(b[lj,li]-b[li,lj]) + a[j,i]*(b[li,lj]-b[lj,li])
          + sum_{k!=i,j} [a[i,k]*(b[lj,p[k]]-b[li,p[k]]) + a[k,i]*(b[p[k],lj]-b[p[k],li])
                         + a[j,k]*(b[li,p[k]]-b[lj,p[k]]) + a[k,j]*(b[p[k],li]-b[p[k],lj])]
    
    Let row_i = a[i,:], row_j = a[j,:], col_i = a[:,i], col_j = a[:,j]
    
    For k != i, j:
    term_k = a[i,k]*(b[lj,p[k]]-b[li,p[k]]) + a[k,i]*(b[p[k],lj]-b[p[k],li])
            + a[j,k]*(b[li,p[k]]-b[lj,p[k]]) + a[k,j]*(b[p[k],li]-b[p[k],lj])
    
    = (a[i,k] - a[j,k])*(b[lj,p[k]]-b[li,p[k]]) + (a[j,k] - a[i,k])*(b[li,p[k]]-b[lj,p[k]]) ... 
    Wait, let me just compute directly.
    
    Actually, let's define:
    b_li = b[li, :]  # row li of b
    b_lj = b[lj, :]  # row lj of b
    b_col_li = b[:, li]
    b_col_lj = b[:, lj]
    
    For k != i,j:
    delta_k = a[i,k]*(b[lj,p[k]] - b[li,p[k]]) + a[k,i]*(b[p[k],lj] - b[p[k],li])
            + a[j,k]*(b[li,p[k]] - b[lj,p[k]]) + a[k,j]*(b[p[k],li] - b[p[k],lj])
    
    = (a[i,k] - a[j,k])*(b[lj,p[k]] - b[li,p[k]]) + (a[k,i] - a[k,j])*(b[p[k],lj] - b[p[k],li])
    ... no that's not right either. Let me just expand:
    
    = a[i,k]*b[lj,p[k]] - a[i,k]*b[li,p[k]] + a[k,i]*b[p[k],lj] - a[k,i]*b[p[k],li]
      + a[j,k]*b[li,p[k]] - a[j,k]*b[lj,p[k]] + a[k,j]*b[p[k],li] - a[k,j]*b[p[k],lj]
    
    = (a[i,k]-a[j,k])*(b[lj,p[k]]-b[li,p[k]]) + (a[k,i]-a[k,j])*(b[p[k],lj]-b[p[k],li])
    
    Hmm, let me verify:
    (a[i,k]-a[j,k])*(b[lj,p[k]]-b[li,p[k]]) = a[i,k]*b[lj,p[k]] - a[i,k]*b[li,p[k]] - a[j,k]*b[lj,p[k]] + a[j,k]*b[li,p[k]]
    (a[k,i]-a[k,j])*(b[p[k],lj]-b[p[k],li]) = a[k,i]*b[p[k],lj] - a[k,i]*b[p[k],li] - a[k,j]*b[p[k],lj] + a[k,j]*b[p[k],li]
    
    Sum = a[i,k]*b[lj,p[k]] - a[i,k]*b[li,p[k]] - a[j,k]*b[lj,p[k]] + a[j,k]*b[li,p[k]]
         + a[k,i]*b[p[k],lj] - a[k,i]*b[p[k],li] - a[k,j]*b[p[k],lj] + a[k,j]*b[p[k],li]
    
    This matches the expanded form. Good.
    """
    li = p[i]
    lj = p[j]
    
    # Diagonal and cross terms
    delta = (a[i, i] - a[j, j]) * (b[lj, lj] - b[li, li])
    delta += (a[i, j] - a[j, i]) * (b[lj, li] - b[li, lj])
    
    # Sum over k != i, j
    # Precompute vectors
    b_lj_p = b[lj, p]  # b[lj, p[k]] for all k
    b_li_p = b[li, p]  # b[li, p[k]] for all k
    b_p_lj = b[p, lj]  # b[p[k], lj] for all k
    b_p_li = b[p, li]  # b[p[k], li] for all k
    
    # a_i_minus_a_j[k] = a[i,k] - a[j,k] for k != i,j
    # a_ki_minus_a_kj[k] = a[k,i] - a[k,j] for k != i,j
    
    diff_row = a[i, :] - a[j, :]
    diff_col = a[:, i] - a[:, j]
    
    delta_b1 = b_lj_p - b_li_p  # b[lj,p[k]] - b[li,p[k]]
    delta_b2 = b_p_lj - b_p_li  # b[p[k],lj] - b[p[k],li]
    
    # Sum over all k, then subtract k=i and k=j
    total = np.dot(diff_row, delta_b1) + np.dot(diff_col, delta_b2)
    
    # Subtract contributions from k=i and k=j
    # For k=i:
    # diff_row[i] = a[i,i]-a[j,i], delta_b1[i] = b[lj,p[i]]-b[li,p[i]] = b[lj,li]-b[li,li]
    # diff_col[i] = a[i,i]-a[i,j], delta_b2[i] = b[p[i],lj]-b[p[i],li] = b[li,lj]-b[li,li]
    total -= diff_row[i] * delta_b1[i] + diff_col[i] * delta_b2[i]
    
    # For k=j:
    # diff_row[j] = a[i,j]-a[j,j], delta_b1[j] = b[lj,p[j]]-b[li,p[j]] = b[lj,lj]-b[li,lj]
    # diff_col[j] = a[j,i]-a[j,j], delta_b2[j] = b[p[j],lj]-b[p[j],li] = b[lj,lj]-b[lj,li]
    total -= diff_row[j] * delta_b1[j] + diff_col[j] * delta_b2[j]
    
    delta += total
    return int(delta)


def _local_search(a, b, p, max_iters=5000, time_limit=None, start_time=None):
    """First-improvement local search with swap moves."""
    n = len(p)
    p = np.array(p, dtype=np.int64)
    current_cost = _qap_cost(a, b, p)
    
    if start_time is None:
        start_time = time.time()
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Try all pairs in a random order or sequential
        # For efficiency, try sequential first-improvement
        for i in range(n):
            if time_limit is not None and (time.time() - start_time) > time_limit:
                break
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    # After a swap, restart the inner loop from beginning for this i
                    # Actually, first-improvement: we can break inner loop and continue
                    break
            if improved:
                continue
    return p.tolist(), current_cost


def _greedy_construct(a, b, n):
    """Greedy construction: assign facilities one by one to the best location."""
    p = np.empty(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    
    # Order facilities by some heuristic (e.g., row sum of a)
    order = np.argsort(-a.sum(axis=1), kind='stable')
    
    for fac in order:
        best_loc = -1
        best_cost = np.inf
        for loc in range(n):
            if used[loc]:
                continue
            # Cost contribution of placing fac at loc, considering already placed facilities
            # sum over already placed k: a[fac,k]*b[loc,p[k]] + a[k,fac]*b[p[k],loc] + a[fac,fac]*b[loc,loc]
            placed = np.where(~used)[0]
            if len(placed) == 0:
                cost = a[fac, fac] * b[loc, loc]
            else:
                # Only consider already placed facilities
                idx = np.where(~used & (np.arange(n) != loc))[0]
                # Actually, we need to check which are already assigned
                assigned_locs = p[~used]  # this doesn't work cleanly
                # Let's just compute incrementally
                cost = a[fac, fac] * b[loc, loc]
                for k in range(n):
                    if not used[k]:
                        continue
                    cost += a[fac, k] * b[loc, p[k]] + a[k, fac] * b[p[k], loc]
            if cost < best_cost:
                best_cost = cost
                best_loc = loc
        p[fac] = best_loc
        used[best_loc] = True
    
    return p.tolist()


def _hungarian_init(a, b, n):
    """Initialize using a linear approximation."""
    # Linearize: minimize sum_i sum_j a[i,j] * b[p[i], p[j]]
    # Approximate by sum_i (row sum of a at i) * (row sum of b at p[i])
    # Or use a different linearization
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Minimize sum_i a_row_sums[i] * b_row_sums[p[i]]
    cost_matrix = np.outer(a_row_sums, b_row_sums)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p.tolist()


def _random_init(n):
    """Random permutation."""
    p = np.random.permutation(n)
    return p.tolist()


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Generate diverse initial solutions
    inits = []
    
    # Rank-matching init
    heavy = np.argsort(-a.sum(axis=1), kind='stable')
    central = np.argsort(b.sum(axis=1), kind='stable')
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    inits.append(p1.tolist())
    
    # Reverse rank-matching
    p2 = np.empty(n, dtype=np.int64)
    p2[heavy] = central[::-1]
    inits.append(p2.tolist())
    
    # Hungarian init
    inits.append(_hungarian_init(a, b, n))
    
    # A few random inits
    for _ in range(3):
        inits.append(_random_init(n))
    
    # Phase 2: Local search from each init
    for init_p in inits:
        if time.time() > deadline - 20:  # Leave time for at least one more LS
            break
        
        p, cost = _local_search(a, b, init_p, max_iters=2000, 
                                  time_limit=deadline - 10, start_time=start_time)
        if cost < best_cost:
            best_cost = cost
            best_p = p
    
    # Phase 3: If time remains, do more local search with perturbations
    if best_p is not None and time.time() < deadline - 30:
        # Simulated-annealing-like restarts
        p = np.array(best_p, dtype=np.int64)
        current_cost = best_cost
        
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            # Perturb: random swaps
            num_swaps = np.random.randint(3, 10)
            for _ in range(num_swaps):
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]
            
            p, cost = _local_search(a, b, p, max_iters=1000,
                                     time_limit=deadline - 5, start_time=start_time)
            if cost < best_cost:
                best_cost = cost
                best_p = p
                p = np.array(best_p, dtype=np.int64)
    
    if best_p is None:
        # Fallback
        best_p = inits[0]
    
    return best_p
# EVOLVE-BLOCK-END
