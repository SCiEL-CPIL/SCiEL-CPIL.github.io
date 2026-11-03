"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist


def _total_cost(a, b, p):
    """Compute the QAP cost for permutation p."""
    # p[i] = location of facility i
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Efficient computation using matrix operations
    # b_perm[i,j] = b[p[i], p[j]]
    b_perm = b[np.ix_(p, p)]
    return int(np.sum(a * b_perm))


def _compute_delta(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in permutation p.
    p[i] = location of facility i, p[j] = location of facility j
    After swap: p[i] <-> p[j]
    """
    li = p[i]
    lj = p[j]
    
    # The cost contribution involving facility i or j changes
    # Original: sum_k a[i,k]*b[li,p[k]] + sum_k a[k,i]*b[p[k],li] 
    #           + sum_k a[j,k]*b[lj,p[k]] + sum_k a[k,j]*b[p[k],lj]
    #           - a[i,j]*b[li,lj] - a[j,i]*b[lj,li]  (counted twice, need to fix)
    
    # Actually, let's think more carefully.
    # Total cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # When we swap positions of i and j:
    # - For all k not in {i,j}: a[i,k]*b[li,p[k]] becomes a[i,k]*b[lj,p[k]], etc.
    # - The terms involving i or j change.
    
    # Let's compute delta directly:
    # delta = new_cost - old_cost
    # = sum_{k} a[i,k]*b[lj,p[k]] + a[k,i]*b[p[k],lj] 
    #   + sum_{k} a[j,k]*b[li,p[k]] + a[k,j]*b[p[k],li]
    #   - sum_{k} a[i,k]*b[li,p[k]] - a[k,i]*b[p[k],li]
    #   - sum_{k} a[j,k]*b[lj,p[k]] - a[k,j]*b[p[k],lj]
    #   + corrections for i,j terms being double counted
    
    # More carefully:
    # Old terms involving i: sum_k a[i,k]*b[li,p[k]] (includes k=j: a[i,j]*b[li,lj])
    # Old terms involving j: sum_k a[j,k]*b[lj,p[k]] (includes k=i: a[j,i]*b[lj,li])
    # Note: a[i,j]*b[li,lj] and a[j,i]*b[lj,li] are in both sums? No.
    # sum_k a[i,k]*b[li,p[k]] includes k=j: a[i,j]*b[li,p[j]] = a[i,j]*b[li,lj]
    # sum_k a[j,k]*b[lj,p[k]] includes k=i: a[j,i]*b[lj,p[i]] = a[j,i]*b[lj,li]
    # These are distinct terms in the total sum.
    
    # New terms involving i (now at lj): sum_k a[i,k]*b[lj,p[k]] (k=j: a[i,j]*b[lj,lj])
    # New terms involving j (now at li): sum_k a[j,k]*b[li,p[k]] (k=i: a[j,i]*b[li,li])
    
    # But wait, when k=i in the new sum for facility i: a[i,i]*b[lj,p[i]] = a[i,i]*b[lj,lj]
    # And when k=j in the new sum for facility j: a[j,j]*b[li,p[j]] = a[j,j]*b[li,li]
    
    # Let me just compute it as:
    # delta = [sum_k a[i,k]*b[lj,p[k]] + sum_k a[k,i]*b[p[k],lj]] 
    #       + [sum_k a[j,k]*b[li,p[k]] + sum_k a[k,j]*b[p[k],li]]
    #       - [sum_k a[i,k]*b[li,p[k]] + sum_k a[k,i]*b[p[k],li]]
    #       - [sum_k a[j,k]*b[lj,p[k]] + sum_k a[k,j]*b[p[k],lj]]
    #       + 2*a[i,j]*b[li,lj] + 2*a[j,i]*b[lj,li]
    #       - 2*a[i,j]*b[lj,lj] - 2*a[j,i]*b[li,li]
    #       ... this is getting complicated. Let me just do it by computing the relevant row/col sums.
    
    # Simpler approach: compute the difference directly
    # For facility i moving from li to lj:
    # Old contribution: sum_k a[i,k]*b[li,p[k]] + sum_k a[k,i]*b[p[k],li] - a[i,i]*b[li,li] - a[i,i]*b[li,li]
    # Wait, the diagonal terms: a[i,i]*b[li,li] appears once in sum_k a[i,k]*b[li,p[k]] (k=i) 
    # and once in sum_k a[k,i]*b[p[k],li] (k=i). So it's counted twice.
    
    # Let me just compute it numerically for the pairs involved.
    # The change only affects terms where i or j is involved.
    
    # Terms with i as first index: a[i,k]*b[p[i],p[k]] for all k
    # Terms with i as second index: a[k,i]*b[p[k],p[i]] for all k
    # Similarly for j.
    
    # Old:
    # old_i = sum_k a[i,k]*b[li,p[k]]  (k ranges over all)
    # old_i2 = sum_k a[k,i]*b[p[k],li]
    # old_j = sum_k a[j,k]*b[lj,p[k]]
    # old_j2 = sum_k a[k,j]*b[p[k],lj]
    
    # New (after swap, i at lj, j at li):
    # new_i = sum_k a[i,k]*b[lj,p[k]]  but p[k] for k!=i,j is same, p[i]=lj, p[j]=li
    #        = sum_{k!=i,j} a[i,k]*b[lj,p[k]] + a[i,i]*b[lj,lj] + a[i,j]*b[lj,li]
    # new_i2 = sum_k a[k,i]*b[p[k],lj]
    #        = sum_{k!=i,j} a[k,i]*b[p[k],lj] + a[i,i]*b[li,lj] + a[j,i]*b[li,lj]
    # new_j = sum_k a[j,k]*b[li,p[k]]
    #        = sum_{k!=i,j} a[j,k]*b[li,p[k]] + a[j,i]*b[li,lj] + a[j,j]*b[li,li]
    # new_j2 = sum_k a[k,j]*b[p[k],li]
    #        = sum_{k!=i,j} a[k,j]*b[p[k],li] + a[i,j]*b[lj,li] + a[j,j]*b[li,li]
    
    # This is complex. Let me use a different approach: 
    # delta = sum over all affected terms.
    
    # Actually, the cleanest way:
    # Let S_i(x) = sum_k a[i,k]*b[x,p[k]]  (row i of a times column x of b_perm)
    # Let T_i(x) = sum_k a[k,i]*b[p[k],x]  (column i of a times row x of b_perm)
    
    # But p changes for i and j. This is still tricky.
    
    # Let me just compute it by brute force for the delta, which for n=100 is O(n) per swap.
    
    # Old cost terms involving i or j:
    # For each k:
    #   a[i,k]*b[li,p[k]] + a[k,i]*b[p[k],li]
    #   a[j,k]*b[lj,p[k]] + a[k,j]*b[p[k],lj]
    # New cost terms involving i or j:
    #   a[i,k]*b[lj,p[k]'] + a[k,i]*b[p[k]',lj]
    #   a[j,k]*b[li,p[k]'] + a[k,j]*b[p[k]',li]
    # where p[k]' = p[k] for k not in {i,j}, p[i]'=lj, p[j]'=li
    
    # For k not in {i,j}: p[k]' = p[k]
    #   old: a[i,k]*b[li,p[k]] + a[k,i]*b[p[k],li] + a[j,k]*b[lj,p[k]] + a[k,j]*b[p[k],lj]
    #   new: a[i,k]*b[lj,p[k]] + a[k,i]*b[p[k],lj] + a[j,k]*b[li,p[k]] + a[k,j]*b[p[k],li]
    #   delta_k = a[i,k]*(b[lj,p[k]]-b[li,p[k]]) + a[k,i]*(b[p[k],lj]-b[p[k],li])
    #            + a[j,k]*(b[li,p[k]]-b[lj,p[k]]) + a[k,j]*(b[p[k],li]-b[p[k],lj])
    #            = (a[i,k]-a[j,k])*(b[lj,p[k]]-b[li,p[k]]) + (a[k,i]-a[k,j])*(b[p[k],lj]-b[p[k],li])
    
    # For k=i:
    #   old: a[i,i]*b[li,li] + a[i,i]*b[li,li] = 2*a[i,i]*b[li,li]
    #         a[j,i]*b[lj,li] + a[i,j]*b[li,lj]
    #   new: a[i,i]*b[lj,lj] + a[i,i]*b[lj,lj] = 2*a[i,i]*b[lj,lj]
    #         a[j,i]*b[li,lj] + a[i,j]*b[lj,li]
    #   delta_i = 2*a[i,i]*(b[lj,lj]-b[li,li]) + a[j,i]*(b[li,lj]-b[lj,li]) + a[i,j]*(b[lj,li]-b[li,lj])
    #            = 2*a[i,i]*(b[lj,lj]-b[li,li]) + (a[j,i]-a[i,j])*(b[li,lj]-b[lj,li])
    
    # For k=j:
    #   old: a[i,j]*b[li,lj] + a[j,i]*b[lj,li]
    #         a[j,j]*b[lj,lj] + a[j,j]*b[lj,lj] = 2*a[j,j]*b[lj,lj]
    #   new: a[i,j]*b[lj,li] + a[j,i]*b[li,lj]
    #         a[j,j]*b[li,li] + a[j,j]*b[li,li] = 2*a[j,j]*b[li,li]
    #   delta_j = a[i,j]*(b[lj,li]-b[li,lj]) + a[j,i]*(b[li,lj]-b[lj,li]) + 2*a[j,j]*(b[li,li]-b[lj,lj])
    #            = (a[i,j]-a[j,i])*(b[lj,li]-b[li,lj]) + 2*a[j,j]*(b[li,li]-b[lj,lj])
    
    # Total delta = sum_{k not in {i,j}} delta_k + delta_i + delta_j
    
    # Let's precompute for efficiency. For a given p, li, lj:
    # Let d_b[k] = b[lj,k] - b[li,k]  (row lj - row li of b)
    # Let d_bT[k] = b[k,lj] - b[k,li]  (col lj - col li of b)
    
    # delta_k for k not in {i,j}:
    #   (a[i,k]-a[j,k])*d_b[p[k]] + (a[k,i]-a[k,j])*d_bT[p[k]]
    
    # This is O(n) per swap evaluation, which for n=100 is fine.
    
    d_b = b[lj, :] - b[li, :]      # shape (n,)
    d_bT = b[:, lj] - b[:, li]     # shape (n,)
    
    # For all k:
    # term1[k] = (a[i,k]-a[j,k]) * d_b[p[k]]
    # term2[k] = (a[k,i]-a[k,j]) * d_bT[p[k]]
    
    ai = a[i, :]
    aj = a[j, :]
    a_i_col = a[:, i]
    a_j_col = a[:, j]
    
    diff_row = ai - aj          # shape (n,)
    diff_col = a_i_col - a_j_col  # shape (n,)
    
    # For k not in {i,j}:
    # We need sum of (diff_row[k]*d_b[p[k]] + diff_col[k]*d_bT[p[k]]) for k not in {i,j}
    
    # Compute for all k first, then subtract k=i and k=j contributions
    all_k = np.arange(len(p))
    pk = p  # p[k] for k in all_k
    
    vals = diff_row * d_b[pk] + diff_col * d_bT[pk]
    
    # Subtract k=i and k=j
    vals_i = diff_row[i] * d_b[pk[i]] + diff_col[i] * d_bT[pk[i]]
    vals_j = diff_row[j] * d_b[pk[j]] + diff_col[j] * d_bT[pk[j]]
    
    sum_not_ij = np.sum(vals) - vals_i - vals_j
    
    # delta_i and delta_j as computed above
    delta_i = 2*a[i,i]*(b[lj,lj]-b[li,li]) + (a[j,i]-a[i,j])*(b[li,lj]-b[lj,li])
    delta_j = (a[i,j]-a[j,i])*(b[lj,li]-b[li,lj]) + 2*a[j,j]*(b[li,li]-b[lj,lj])
    
    return int(sum_not_ij + delta_i + delta_j)


def _steepest_descent(a, b, p, max_iters=10000, time_limit=10.0):
    """
    Steepest descent local search: at each step, find the best swap and apply it.
    """
    n = len(p)
    p = p.copy()
    current_cost = _total_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        best_delta = 0
        best_i = -1
        best_j = -1
        
        for i in range(n):
            for j in range(i+1, n):
                delta = _compute_delta(a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_delta >= 0:
            break
        
        # Apply the swap
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    return best_p, best_cost


def _random_restart(a, b, p0, n_restarts=5, time_per_restart=15.0):
    """
    Multiple restarts with random perturbations followed by steepest descent.
    """
    n = len(p)
    best_p = p0.copy()
    best_cost = _total_cost(a, b, p0)
    
    start_time = time.time()
    
    for r in range(n_restarts):
        if time.time() - start_time > time_per_restart * r:
            break
        
        # Create a perturbed version
        p = p0.copy()
        # Randomly swap a few pairs
        n_swaps = max(1, n // 10)
        for _ in range(n_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        
        p, cost = _steepest_descent(a, b, p, max_iters=5000, 
                                     time_limit=min(10.0, time_per_restart - (time.time() - start_time)))
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    total_deadline = 140.0  # safety margin before 150s kill
    start_time = time.time()
    
    # Initial solution: rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # First pass: steepest descent on the initial solution
    p, cost = _steepest_descent(a, b, p, max_iters=20000, 
                                 time_limit=min(30.0, total_deadline - (time.time() - start_time)))
    best_p = p.copy()
    best_cost = cost
    
    # Second phase: multiple random restarts
    remaining = total_deadline - (time.time() - start_time)
    if remaining > 5:
        # Use the best solution found as the base for restarts
        p, cost = _random_restart(a, b, best_p, n_restarts=10, 
                                   time_per_restart=min(15.0, remaining - 2.0))
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    # Final check: make sure we have a valid permutation
    assert len(best_p) == n
    assert set(best_p.tolist()) == set(range(n))
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
