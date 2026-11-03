"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute the QAP cost for a given permutation p."""
    # p[i] is the location for facility i
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # This is equivalent to sum of (a * b[np.ix_(p, p)])
    bp = b[np.ix_(p, p)]
    return np.sum(a * bp)


def _greedy_initial(a, b):
    """
    Greedy construction: place facilities one by one, choosing the location
    that minimizes the incremental cost.
    """
    n = len(a)
    p = np.empty(n, dtype=np.int64)
    placed = np.zeros(n, dtype=bool)
    # Track which locations are taken
    loc_taken = np.zeros(n, dtype=bool)
    
    # Precompute row sums of a for ordering
    a_row_sums = a.sum(axis=1)
    order = np.argsort(-a_row_sums, kind='stable')
    
    # For incremental cost calculation, we need:
    # When placing facility i at location l, the incremental cost is:
    # sum_{j placed} a[i,j] * b[l, p[j]] + sum_{j placed} a[j,i] * b[p[j], l]
    # + sum_{j placed} a[i,j] * b[p[j], l] ... wait, let me think more carefully.
    
    # Actually, the total cost is sum_{i,j} a[i,j] * b[p[i], p[j]]
    # When we place facility i at location l, the new terms involving i are:
    # For each already-placed facility j at location p[j]:
    #   a[i,j] * b[l, p[j]]  (term where first index is i)
    #   a[j,i] * b[p[j], l]  (term where second index is i)
    # Plus a[i,i] * b[l, l] if we include self (but usually i != j in QAP)
    
    # Let's just compute incrementally.
    # Precompute: for each facility i, we need to know the interaction with placed facilities.
    
    for idx in order:
        best_loc = -1
        best_cost = np.inf
        for l in range(n):
            if loc_taken[l]:
                continue
            # Compute incremental cost of placing idx at l
            # Sum over all already placed j:
            #   a[idx, j] * b[l, p[j]] + a[j, idx] * b[p[j], l]
            # We can vectorize this
            if not np.any(placed):
                inc = 0.0
            else:
                placed_mask = placed
                # a[idx, placed] * b[l, p[placed]]
                term1 = np.dot(a[idx, placed_mask], b[l, p[placed_mask]])
                # a[placed, idx] * b[p[placed], l]
                term2 = np.dot(a[placed_mask, idx], b[p[placed_mask], l])
                inc = term1 + term2
            if inc < best_cost:
                best_cost = inc
                best_loc = l
        p[idx] = best_loc
        placed[idx] = True
        loc_taken[best_loc] = True
    
    return p


def _local_search_2opt(a, b, p, time_limit=None, deadline=None):
    """
    2-opt local search: swap two locations in the permutation and accept if cost decreases.
    Uses incremental cost update for efficiency.
    """
    n = len(a)
    p = p.copy()
    cost = _qap_cost(a, b, p)
    
    # Precompute for faster incremental updates
    # When swapping locations of facilities i and j (i.e., p[i] and p[j] are swapped),
    # the change in cost can be computed efficiently.
    
    # Let's precompute some useful quantities
    # For a swap of p[i] and p[j]:
    # Old contribution involving i and j:
    #   For all k: a[i,k]*b[p[i],p[k]] + a[k,i]*b[p[k],p[i]] 
    #             + a[j,k]*b[p[j],p[k]] + a[k,j]*b[p[k],p[j]]
    #   minus a[i,j]*b[p[i],p[j]] - a[j,i]*b[p[j],p[i]] (counted twice in above)
    # This is getting complex. Let's use a simpler approach:
    # Compute the delta directly.
    
    # Actually, let's use the standard 2-opt delta formula for QAP.
    # When we swap the locations of facility i and facility j:
    # Let li = p[i], lj = p[j]
    # The change in cost is:
    # delta = sum_{k != i, j} [a[i,k]*b[li,p[k]] + a[k,i]*b[p[k],li] + a[j,k]*b[lj,p[k]] + a[k,j]*b[p[k],lj]]
    #         - sum_{k != i, j} [a[i,k]*b[lj,p[k]] + a[k,i]*b[p[k],lj] + a[j,k]*b[li,p[k]] + a[k,j]*b[p[k],li]]
    #         + a[i,j]*b[li,lj] + a[j,i]*b[lj,li]
    #         - a[i,j]*b[lj,li] - a[j,i]*b[li,lj]
    #         + a[i,i]*b[li,li] + a[j,j]*b[lj,lj]  (these don't change, so cancel)
    #         - a[i,i]*b[lj,lj] - a[j,j]*b[li,li]  (wait, this isn't right either)
    
    # Let me just do it more carefully. The total cost is C = sum_{u,v} a[u,v] * b[p[u], p[v]]
    # After swapping p[i] and p[j], let's call the new permutation p'.
    # p'[i] = lj, p'[j] = li, p'[k] = p[k] for k != i,j
    # 
    # C' - C = sum_{u,v} a[u,v] * (b[p'[u], p'[v]] - b[p[u], p[v]])
    # The only terms that change are those where u or v is i or j.
    # 
    # Terms with u=i, v=i: a[i,i]*(b[li,li] - b[li,li]) = 0 (no change since p'[i]=lj? wait no)
    # Wait, p'[i] = lj, so b[p'[i], p'[i]] = b[lj, lj], and b[p[i], p[i]] = b[li, li]. So change is a[i,i]*(b[lj,lj]-b[li,li])
    # Similarly for u=j,v=j: a[j,j]*(b[li,li]-b[lj,lj])
    # For u=i, v=j: a[i,j]*(b[lj,li]-b[li,lj])
    # For u=j, v=i: a[j,i]*(b[li,lj]-b[lj,li])
    # For u=i, v=k (k!=i,j): a[i,k]*(b[lj,p[k]]-b[li,p[k]])
    # For u=k, v=i (k!=i,j): a[k,i]*(b[p[k],lj]-b[p[k],li])
    # For u=j, v=k (k!=i,j): a[j,k]*(b[li,p[k]]-b[lj,p[k]])
    # For u=k, v=j (k!=i,j): a[k,j]*(b[p[k],li]-b[p[k],lj])
    
    # So delta = a[i,i]*(b[lj,lj]-b[li,li]) + a[j,j]*(b[li,li]-b[lj,lj])
    #            + a[i,j]*(b[lj,li]-b[li,lj]) + a[j,i]*(b[li,lj]-b[lj,li])
    #            + sum_{k!=i,j} [a[i,k]*(b[lj,p[k]]-b[li,p[k]]) + a[k,i]*(b[p[k],lj]-b[p[k],li])
    #                              + a[j,k]*(b[li,p[k]]-b[lj,p[k]]) + a[k,j]*(b[p[k],li]-b[p[k],lj])]
    
    # Let's precompute row-wise differences to speed this up.
    # For a given pair (li, lj), define:
    #   diff_i[k] = b[lj, p[k]] - b[li, p[k]]  for the a[i,k] term
    #   diff_ki[k] = b[p[k], lj] - b[p[k], li]  for the a[k,i] term
    # etc.
    
    # This is O(n) per pair, and O(n^2) pairs, so O(n^3) total which is too slow for n=256.
    # n=256, n^3 = 16.7M, which might be okay in NumPy if vectorized well.
    
    # Let's try a vectorized approach for the inner sum.
    
    # Precompute: for each pair of locations (li, lj), we can precompute:
    #   D1[li, lj, k] = b[lj, p[k]] - b[li, p[k]]  -- this is too much memory
    # Instead, let's compute on the fly but vectorize over k.
    
    best_improvement = 0
    no_improve_count = 0
    start_time = time.time()
    
    while True:
        if deadline is not None and time.time() > deadline:
            break
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        found_improvement = False
        best_delta = 0
        best_i = -1
        best_j = -1
        
        for i in range(n):
            li = p[i]
            for j in range(i + 1, n):
                lj = p[j]
                # Compute delta for swapping i and j
                # Self terms
                delta = (a[i, i] * (b[lj, lj] - b[li, li]) +
                         a[j, j] * (b[li, li] - b[lj, lj]) +
                         a[i, j] * (b[lj, li] - b[li, lj]) +
                         a[j, i] * (b[li, lj] - b[lj, li]))
                
                # Cross terms with all k != i, j
                # We need: sum_{k!=i,j} [a[i,k]*(b[lj,p[k]]-b[li,p[k]]) + a[k,i]*(b[p[k],lj]-b[p[k],li])
                #                          + a[j,k]*(b[li,p[k]]-b[lj,p[k]]) + a[k,j]*(b[p[k],li]-b[p[k],lj])]
                
                # Vectorize over k
                # Create mask for k != i, j
                # To avoid creating masks repeatedly, let's compute for all k and subtract i,j contributions
                # 
                # term1 = a[i,:] * (b[lj, p] - b[li, p])
                # term2 = a[:,i] * (b[p, lj] - b[p, li])
                # term3 = a[j,:] * (b[li, p] - b[lj, p])
                # term4 = a[:,j] * (b[p, li] - b[p, lj])
                # 
                # But we need to exclude k=i and k=j.
                # 
                # Let's compute the full sums and then subtract the i and j contributions.
                
                diff1 = b[lj, p] - b[li, p]  # shape (n,)
                diff2 = b[p, lj] - b[p, li]  # shape (n,)
                diff3 = b[li, p] - b[lj, p]  # = -diff1
                diff4 = b[p, li] - b[p, lj]  # = -diff2
                
                s1 = np.dot(a[i, :], diff1)
                s2 = np.dot(a[:, i], diff2)
                s3 = np.dot(a[j, :], diff3)
                s4 = np.dot(a[:, j], diff4)
                
                # Subtract contributions from k=i and k=j
                # For k=i:
                s1 -= a[i, i] * diff1[i]
                s2 -= a[i, i] * diff2[i]
                s3 -= a[j, i] * diff3[i]
                s4 -= a[i, j] * diff4[i]
                # For k=j:
                s1 -= a[i, j] * diff1[j]
                s2 -= a[j, i] * diff2[j]
                s3 -= a[j, j] * diff3[j]
                s4 -= a[j, j] * diff4[j]
                
                cross_sum = s1 + s2 + s3 + s4
                delta += cross_sum
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_delta < -1e-9:  # significant improvement
            # Perform the swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            cost += best_delta
            found_improvement = True
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= 3:
                break
    
    return p, cost


def solve(a, b):
    """
    Solve the QAP instance using greedy initialization followed by 2-opt local search.
    """
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Greedy initial solution
    p = _greedy_initial(a, b)
    cost = _qap_cost(a, b, p)
    
    # 2-opt local search
    p, cost = _local_search_2opt(a, b, p, deadline=deadline)
    
    # If we have time left, try a few random restarts with 2-opt
    best_p = p.copy()
    best_cost = cost
    
    while time.time() < deadline - 5.0:  # Leave 5s margin
        # Random restart
        p_rand = np.random.permutation(n)
        p_rand, cost_rand = _local_search_2opt(a, b, p_rand, deadline=deadline)
        if cost_rand < best_cost:
            best_p = p_rand
            best_cost = cost_rand
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
