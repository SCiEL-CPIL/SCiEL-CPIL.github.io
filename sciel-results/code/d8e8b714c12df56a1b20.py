"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute the QAP cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _greedy_initial(a, b):
    """
    Greedy assignment: place facilities one by one choosing the location
    that minimizes the marginal cost given already-placed facilities.
    Start with the facility having the largest row-sum in a.
    """
    n = len(a)
    p = np.full(n, -1, dtype=np.int64)
    # Precompute: for each facility i, the contribution if placed at location l
    # given already placed facilities.
    
    # Order facilities by descending total flow (row sum of a)
    order = np.argsort(-a.sum(axis=1), kind='stable')
    
    placed = np.zeros(n, dtype=bool)
    # marginal_cost[i, l] = sum over already placed j of a[i,j]*b[l, p[j]] + a[j,i]*b[p[j], l]
    # We'll compute incrementally.
    
    # For efficiency, maintain for each location l the accumulated cost contribution
    # acc[l] = sum over placed j of (a[i,j]*b[l,p[j]] for current i) -- but this depends on i.
    # Better: maintain dist_to_placed[l, j] style... too expensive.
    
    # Alternative: maintain for each location l, the vector of distances to placed facilities,
    # weighted by flows. Actually, let's just do a simple greedy with O(n^2) per step = O(n^3) total.
    # n=256, n^3 = 16.7M, which is fine.
    
    for idx in range(n):
        i = order[idx]
        best_l = -1
        best_cost = np.inf
        for l in range(n):
            if placed[l]:
                continue
            # Cost of placing facility i at location l
            cost = 0
            for j in range(n):
                if placed[j]:
                    cost += a[i, j] * b[l, p[j]] + a[j, i] * b[p[j], l]
            # Also include self-term if any (a[i,i]*b[l,l])
            cost += a[i, i] * b[l, l]
            if cost < best_cost:
                best_cost = cost
                best_l = l
        p[i] = best_l
        placed[best_l] = True
    
    return p


def _local_search_2opt(a, b, p, max_time=None, start_time=None):
    """
    2-opt style local search: repeatedly swap two facilities' locations
    if it reduces the cost. Use efficient delta computation.
    """
    if start_time is None:
        start_time = time.time()
    if max_time is None:
        max_time = 140.0  # safety margin
    
    n = len(a)
    current_cost = _qap_cost(a, b, p)
    
    # Precompute for faster delta evaluation
    # For a swap of facilities i and j (at locations p[i] and p[j]):
    # delta = sum_{k != i,j} a[i,k]*(b[p[i],p[k]] - b[p[j],p[k]]) 
    #       + sum_{k != i,j} a[j,k]*(b[p[j],p[k]] - b[p[i],p[k]])
    #       + a[i,j]*(b[p[i],p[j]] - b[p[j],p[i]])  -- but b is symmetric? Not necessarily.
    #       + a[j,i]*(b[p[j],p[i]] - b[p[i],p[j]])
    # Plus the self terms change: a[i,i]*(b[p[i],p[i]] - b[p[j],p[j]])
    #                            + a[j,j]*(b[p[j],p[j]] - b[p[i],p[i]])
    
    # Actually, let's be more careful. Let's define:
    # c(i, l) = sum_k a[i,k] * b[l, p[k]]  (cost contribution of facility i at location l, 
    #   but this double counts with other facilities)
    
    # The total cost is sum_{i,j} a[i,j] * b[p[i], p[j]].
    # If we swap i and j, only terms involving i or j change.
    # Terms involving i: sum_j a[i,j]*b[p[i],p[j]] + sum_j a[j,i]*b[p[j],p[i]]
    # After swap: sum_j a[i,j]*b[p[j],p[j]] ... no wait, p[i] and p[j] swap.
    
    # Let's just compute delta directly for a pair (i, j) with locations li=p[i], lj=p[j]:
    # Old contribution involving i or j:
    #   sum_{k} a[i,k]*b[li,p[k]] + sum_{k} a[k,i]*b[p[k],li]
    #   sum_{k} a[j,k]*b[lj,p[k]] + sum_{k} a[k,j]*b[p[k],lj]
    #   minus the double-counted terms where k=i or k=j... actually the sum is over all k.
    
    # This is getting complex. Let's use a simpler but still efficient approach:
    # Precompute row_sums[i, l] = sum_k a[i,k] * b[l, p[k]] for each facility i and location l.
    # Then the cost of placing i at l (ignoring interaction with j) is row_sums[i, l].
    # But when both i and j are considered, we need to subtract the cross terms.
    
    # For now, let's do a straightforward delta computation with precomputed arrays.
    
    # Precompute: for each location l, and each facility i, 
    #   contrib[i, l] = sum_k a[i, k] * b[l, p[k]]
    # This is n x n matrix.
    
    # Actually, let's just do a simple 2-opt with direct cost recomputation for the affected pairs.
    # For n=256, checking all pairs is 256*255/2 = 32640 pairs per sweep.
    # For each pair, we compute delta in O(n) time. Total per sweep: O(n^3) = 16.7M ops.
    # With numpy, this should be fast enough for several sweeps.
    
    # Let's precompute the matrix B_p = b[np.ix_(p, p)] for quick access.
    B_p = b[np.ix_(p, p)]
    
    improved = True
    no_improve_count = 0
    while improved:
        improved = False
        # Check time
        if time.time() - start_time > max_time:
            break
        
        # Try all pairs
        for i in range(n - 1):
            for j in range(i + 1, n):
                li = p[i]
                lj = p[j]
                
                # Compute delta for swapping i and j
                # Old cost terms involving i or j:
                # sum_k a[i,k]*B_p[i,k] + sum_k a[k,i]*B_p[k,i]  (but B_p[k,i] = b[p[k], p[i]])
                # Actually, the total cost is sum_{i,j} a[i,j]*B_p[i,j].
                # Terms that change: those where first index is i or j, or second index is i or j.
                
                # Let's compute the old contribution of i and j:
                old_ii = a[i, i] * B_p[i, i]
                old_ij = a[i, j] * B_p[i, j]
                old_ji = a[j, i] * B_p[j, i]
                old_jj = a[j, j] * B_p[j, j]
                
                # Cross terms with other k:
                # sum_{k != i, j} a[i,k] * B_p[i,k] + a[k,i] * B_p[k,i]
                # sum_{k != i, j} a[j,k] * B_p[j,k] + a[k,j] * B_p[k,j]
                
                # New positions: i goes to lj, j goes to li.
                # New B_p values for the swapped entries:
                # B_p_new[i, k] = b[lj, p[k]] for k != j
                # B_p_new[k, i] = b[p[k], lj] for k != j
                # B_p_new[j, k] = b[li, p[k]] for k != i
                # B_p_new[k, j] = b[p[k], li] for k != i
                # B_p_new[i, j] = b[lj, li]
                # B_p_new[j, i] = b[li, lj]
                # B_p_new[i, i] = b[lj, lj]
                # B_p_new[j, j] = b[li, li]
                
                # This is getting very complex. Let's just recompute the cost for the 
                # affected submatrix.
                
                # Simpler: compute delta by summing over all k != i, j
                delta = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    # Old: a[i,k]*b[li,p[k]] + a[k,i]*b[p[k],li] + a[j,k]*b[lj,p[k]] + a[k,j]*b[p[k],lj]
                    # New: a[i,k]*b[lj,p[k]] + a[k,i]*b[p[k],lj] + a[j,k]*b[li,p[k]] + a[k,j]*b[p[k],li]
                    delta += a[i, k] * (b[lj, p[k]] - b[li, p[k]])
                    delta += a[k, i] * (b[p[k], lj] - b[p[k], li])
                    delta += a[j, k] * (b[li, p[k]] - b[lj, p[k]])
                    delta += a[k, j] * (b[p[k], li] - b[p[k], lj])
                
                # Self and cross terms between i and j:
                # Old: a[i,i]*b[li,li] + a[i,j]*b[li,lj] + a[j,i]*b[lj,li] + a[j,j]*b[lj,lj]
                # New: a[i,i]*b[lj,lj] + a[i,j]*b[lj,li] + a[j,i]*b[li,lj] + a[j,j]*b[li,li]
                delta += a[i, i] * (b[lj, lj] - b[li, li])
                delta += a[i, j] * (b[lj, li] - b[li, lj])
                delta += a[j, i] * (b[li, lj] - b[lj, li])
                delta += a[j, j] * (b[li, li] - b[lj, lj])
                
                if delta < 0:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    # Update B_p
                    B_p[i, :] = b[lj, p]
                    B_p[:, i] = b[p, lj]
                    B_p[j, :] = b[li, p]
                    B_p[:, j] = b[p, li]
                    B_p[i, i] = b[lj, lj]
                    B_p[j, j] = b[li, li]
                    B_p[i, j] = b[lj, li]
                    B_p[j, i] = b[li, lj]
        
        if not improved:
            break
    
    return p, current_cost


def solve(a, b):
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    n = len(a)
    
    # Generate initial solution using greedy
    p = _greedy_initial(a, b)
    
    # Run local search
    p, cost = _local_search_2opt(a, b, p, max_time=deadline, start_time=start_time)
    
    # Check if we have time for another pass
    if time.time() - start_time < deadline - 10:
        p, cost = _local_search_2opt(a, b, p, max_time=deadline, start_time=start_time)
    
    return p.tolist()
# EVOLVE-BLOCK-END
