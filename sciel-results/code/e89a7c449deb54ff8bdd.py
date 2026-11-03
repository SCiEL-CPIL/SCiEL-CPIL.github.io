"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute the QAP objective sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return np.sum(a * b[np.ix_(p, p)])


def _greedy_initial(a, b):
    """
    Greedy construction: place facilities one by one, choosing the location
    that minimizes the incremental cost contribution.
    
    When placing facility i at location l, the incremental cost is:
      sum over already-placed facilities k: a[i,k]*b[l, p[k]] + a[k,i]*b[p[k], l]
    Plus the self-term a[i,i]*b[l,l].
    
    We maintain an array of incremental costs for each unplaced facility at each location.
    """
    n = len(a)
    p = np.full(n, -1, dtype=np.int64)
    placed = np.zeros(n, dtype=bool)
    
    # Precompute row sums of a for efficiency
    # For facility i, the cost of placing at location l given current placement:
    # cost_i(l) = a[i,i]*b[l,l] + sum_{k: placed} (a[i,k]*b[l,p[k]] + a[k,i]*b[p[k],l])
    
    # We'll maintain for each unplaced facility i, an array of costs for each location
    # But that's O(n^2) memory and updates. Let's do it iteratively.
    
    # Start: no facilities placed. The "cost" of placing facility i at location l is just a[i,i]*b[l,l].
    # But actually the full cost includes all pairs. Let's think differently.
    
    # The total cost is sum_{i,j} a[i,j] * b[p[i], p[j]].
    # When we add facility i at location l, the new terms involving i are:
    #   sum_j a[i,j] * b[l, p[j]]  (where p[j] is known for placed j)
    #   sum_j a[j,i] * b[p[j], l]  (where p[j] is known for placed j)
    #   a[i,i] * b[l, l]
    
    # So incremental cost for placing i at l = a[i,i]*b[l,l] + sum_{k placed} (a[i,k]*b[l,p[k]] + a[k,i]*b[p[k],l])
    
    # To make this efficient, maintain:
    #   col_sum[k] = b[p[k], :] ... no, let's maintain vectors.
    
    # For each unplaced facility i, define:
    #   inc_i(l) = a[i,i]*b[l,l] + sum_{k placed} (a[i,k]*b[l,p[k]] + a[k,i]*b[p[k],l])
    
    # We can maintain two arrays:
    #   B_rows: for placed locations, we have b[p[k], :] for each k
    #   B_cols: for placed locations, we have b[:, p[k]] for each k
    #   a_rows[i, k] = a[i, k], a_cols[i, k] = a[k, i]
    
    # inc_i(l) = a[i,i]*b[l,l] + sum_{k placed} a[i,k]*b[l,p[k]] + sum_{k placed} a[k,i]*b[p[k],l]
    #           = a[i,i]*b[l,l] + (a[i, :][:, placed] @ b[l, p[placed]]) + (a[:, i][:, placed] @ b[p[placed], l])
    
    # Let's maintain:
    #   b_rows_placed: shape (num_placed, n) = b[p[placed], :]
    #   b_cols_placed: shape (num_placed, n) = b[:, p[placed]]
    #   a_row_sums: for each unplaced i, a[i, placed] @ b_rows_placed.T ... 
    
    # Simpler: just compute incrementally. n=256, so O(n^3) greedy might be too slow.
    # Let's estimate: n=256, each step we compute costs for all unplaced facilities at all locations.
    # That's O((n-t)^2 * t) per step, summing to O(n^4/3) ~ 256^4/3 ~ 1.4e9. Too slow.
    
    # Better approach: maintain the incremental cost matrix incrementally.
    # Let C[i, l] = incremental cost of placing facility i at location l, given current placement.
    # Initially (no placement): C[i, l] = a[i,i] * b[l,l]
    # When we place facility k at location l_k:
    #   For each unplaced i: C[i, l] += a[i,k]*b[l, l_k] + a[k,i]*b[l_k, l]
    #   This is O(n^2) per placement, total O(n^3) = 256^3 ~ 1.7e7. Fine.
    
    C = a.diagonal()[:, None] * b.diagonal()[None, :].astype(np.float64)
    
    for _ in range(n):
        # Find the (i, l) pair with minimum C[i, l] where i is unplaced and l is unplaced
        # Mask placed facilities and locations
        mask_i = ~placed
        # We need to find min over unplaced i and unplaced l
        # Set placed locations to inf
        C_masked = C.copy()
        C_masked[~mask_i, :] = np.inf
        # Also mask placed locations (columns)
        # We don't have a direct column mask, but we can track which locations are used
        # Let's track used locations
        # Actually, let's just track both.
        
        # Find argmin
        idx = np.unravel_index(np.argmin(C_masked), C_masked.shape)
        i, l = idx[0], idx[1]
        
        p[i] = l
        placed[i] = True
        
        # Update C for all unplaced facilities
        # C[i', l'] += a[i', i] * b[l', l] + a[i, i'] * b[l, l']
        # Only for unplaced i'
        unplaced = np.where(~placed)[0]
        if len(unplaced) > 0:
            # b[l', l] for all l' -> b[:, l]
            # b[l, l'] for all l' -> b[l, :]
            C[unplaced, :] += a[unplaced, i][:, None] * b[:, l][None, :] + a[i, unplaced][None, :] * b[l, :][None, :]
            # Zero out row for placed facility i
            C[i, :] = np.inf
    
    return p


def _local_search_2opt(a, b, p, max_time=100.0):
    """
    2-opt local search: try swapping two locations in the permutation.
    For a swap of p[i] and p[j], the change in cost can be computed efficiently.
    """
    n = len(a)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    deadline = start_time + max_time
    
    # Precompute for efficiency
    # When we swap p[i] and p[j], let's call the new permutation p'.
    # The change in cost:
    # Terms involving i or j change.
    # Let's compute delta for swapping positions i and j (i.e., swapping facilities at these positions).
    # Actually, p[i] is the location of facility i. So swapping p[i] and p[j] means
    # facility i goes to old p[j] and facility j goes to old p[i].
    
    # The cost terms that change are those involving i or j.
    # Let's derive the delta.
    
    # Current contribution of i and j to cost:
    # sum_k a[i,k]*b[p[i],p[k]] + sum_k a[k,i]*b[p[k],p[i]] + sum_k a[j,k]*b[p[j],p[k]] + sum_k a[k,j]*b[p[k],p[j]]
    # But terms where k=i or k=j are counted twice in the full sum? No, the full sum is over all i,j.
    # Let's just compute the full cost change.
    
    # After swap, p'[i] = p[j], p'[j] = p[i], others same.
    # The cost is sum_{u,v} a[u,v] * b[p'[u], p'[v]].
    # Terms that change are those where u or v is i or j.
    
    # Let's compute delta = new_cost - old_cost by considering all pairs (u,v) where u in {i,j} or v in {i,j}.
    
    # This is getting complex. Let's just recompute the cost after swap for candidate swaps,
    # but that's O(n^2) per swap evaluation, and we have O(n^2) swaps, so O(n^4) total. Too slow.
    
    # Better: precompute the contribution of each facility to the cost.
    # Let's define for each facility i:
    #   row_contrib[i] = sum_j a[i,j] * b[p[i], p[j]]
    #   col_contrib[i] = sum_j a[j,i] * b[p[j], p[i]]
    # Total cost = sum_i (row_contrib[i] + col_contrib[i]) ... no, that double counts.
    # Actually total cost = sum_{i,j} a[i,j]*b[p[i],p[j]].
    # row_contrib[i] = sum_j a[i,j]*b[p[i],p[j]]
    # So total cost = sum_i row_contrib[i].
    
    # When we swap p[i] and p[j]:
    #   New row_contrib[i] = sum_k a[i,k]*b[p[j], p[k]]  (where p[k] is old for k not i,j)
    #   New row_contrib[j] = sum_k a[j,k]*b[p[i], p[k]]
    #   For k not i,j: row_contrib[k] changes because b[p[k], p[i]] and b[p[k], p[j]] change.
    #   Specifically, row_contrib[k] had terms a[k,i]*b[p[k],p[i]] and a[k,j]*b[p[k],p[j]].
    #   After swap, these become a[k,i]*b[p[k],p[j]] and a[k,j]*b[p[k],p[i]].
    
    # This is still complex. Let's use a different approach: 
    # Precompute B = b[p[:, None], p[None, :]] i.e., B[i,j] = b[p[i], p[j]].
    # Then cost = sum(a * B).
    # When we swap p[i] and p[j], B changes in rows i, j and columns i, j.
    # Specifically, B[i, :] and B[:, i] change, and B[j, :] and B[:, j] change.
    # The new B can be computed, and delta = sum(a * (B_new - B_old)).
    
    # Let's maintain B explicitly.
    B = b[np.ix_(p, p)]
    
    # For a swap of i and j:
    #   B_new[i, k] = B[j, k] for all k (since p_new[i] = p_old[j])
    #   B_new[k, i] = B[k, j] for all k
    #   B_new[j, k] = B[i, k] for all k
    #   B_new[k, j] = B[k, i] for all k
    #   Other entries unchanged.
    
    # Delta = sum_{u,v} a[u,v] * (B_new[u,v] - B_old[u,v])
    # Only terms where u in {i,j} or v in {i,j} change.
    
    # Let's compute delta efficiently:
    # delta = sum_k a[i,k]*(B[j,k]-B[i,k]) + sum_k a[k,i]*(B[k,j]-B[k,i]) 
    #       + sum_k a[j,k]*(B[i,k]-B[j,k]) + sum_k a[k,j]*(B[k,i]-B[k,j])
    #       - 4*a[i,j]*(B[i,j]-B[j,i]) ... wait, need to be careful with double counting.
    
    # Actually, let's just compute it directly:
    # delta = (sum_k a[i,k]*B_new[i,k] + sum_k a[k,i]*B_new[k,i] + sum_k a[j,k]*B_new[j,k] + sum_k a[k,j]*B_new[k,j])
    #         - (sum_k a[i,k]*B_old[i,k] + sum_k a[k,i]*B_old[k,i] + sum_k a[j,k]*B_old[j,k] + sum_k a[k,j]*B_old[k,j])
    # But this double counts terms where (u,v) = (i,j) or (j,i) etc.
    
    # Simpler: just compute the full cost change by updating B and recomputing.
    # Since n=256, let's try a different strategy: random restarts with 2-opt.
    
    # Actually, let's just do a simple 2-opt: for each pair (i,j), compute the delta by
    # temporarily swapping and recomputing the affected terms.
    
    # For efficiency, let's precompute:
    #   For each i, row_a[i] = a[i, :]
    #   For each i, col_a[i] = a[:, i]
    
    # When swapping i and j:
    #   delta = sum_k [a[i,k]*(B[j,k]-B[i,k]) + a[k,i]*(B[k,j]-B[k,i]) + a[j,k]*(B[i,k]-B[j,k]) + a[k,j]*(B[k,i]-B[k,j])]
    #   But we need to subtract the double-counted terms.
    
    # Let's just implement it and test.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                # Compute delta for swapping i and j
                # B_old[i, :] -> B_new[i, :] = B_old[j, :]
                # B_old[:, i] -> B_new[:, i] = B_old[:, j]
                # B_old[j, :] -> B_new[j, :] = B_old[i, :]
                # B_old[:, j] -> B_new[:, j] = B_old[:, i]
                
                # delta = sum_{u,v} a[u,v] * (B_new[u,v] - B_old[u,v])
                # Only (i, *), (*, i), (j, *), (*, j) change.
                
                # Let's compute:
                # old_terms = sum_k a[i,k]*B[i,k] + sum_k a[k,i]*B[k,i] + sum_k a[j,k]*B[j,k] + sum_k a[k,j]*B[k,j]
                # But this counts (i,i), (i,j), (j,i), (j,j) multiple times.
                
                # Better: 
                # delta = [sum_k a[i,k]*(B[j,k]-B[i,k]) + sum_k a[k,i]*(B[k,j]-B[k,i]) 
                #        + sum_k a[j,k]*(B[i,k]-B[j,k]) + sum_k a[k,j]*(B[k,i]-B[k,j])]
                #        - correction for double-counted terms
                
                # The terms (i,j) and (j,i) are counted in both the i-row and j-col, etc.
                # Let's just compute it directly by extracting the relevant submatrices.
                
                # Actually, the cleanest way: 
                # delta = sum_{u in {i,j}, v} a[u,v]*(B_new[u,v]-B_old[u,v]) + sum_{v in {i,j}, u not in {i,j}} a[u,v]*(B_new[u,v]-B_old[u,v])
                
                # For u=i: B_new[i,v] = B_old[j,v], so delta_i = sum_v a[i,v]*(B[j,v]-B[i,v])
                # For u=j: B_new[j,v] = B_old[i,v], so delta_j = sum_v a[j,v]*(B[i,v]-B[j,v])
                # For v=i, u not in {i,j}: B_new[u,i] = B_old[u,j], so delta_vi = sum_{u not in {i,j}} a[u,i]*(B[u,j]-B[u,i])
                # For v=j, u not in {i,j}: B_new[u,j] = B_old[u,i], so delta_vj = sum_{u not in {i,j}} a[u,j]*(B[u,i]-B[u,j])
                
                # Total delta = delta_i + delta_j + delta_vi + delta_vj
                
                delta_i = np.dot(a[i, :], B[j, :] - B[i, :])
                delta_j = np.dot(a[j, :], B[i, :] - B[j, :])
                
                # For delta_vi and delta_vj, exclude u=i and u=j
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                mask[j] = False
                delta_vi = np.dot(a[mask, i], B[mask, j] - B[mask, i])
                delta_vj = np.dot(a[mask, j], B[mask, i] - B[mask, j])
                
                delta = delta_i + delta_j + delta_vi + delta_vj
                
                if delta < -1e-9:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    # Update B
                    B[i, :], B[j, :] = B[j, :].copy(), B[i, :].copy()
                    B[:, i], B[:, j] = B[:, j].copy(), B[:, i].copy()
                    # Fix the (i,j) and (j,i) entries which were swapped twice
                    B[i, j] = b[p[i], p[j]]
                    B[j, i] = b[p[j], p[i]]
                    B[i, i] = b[p[i], p[i]]
                    B[j, j] = b[p[j], p[j]]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Phase 1: Greedy construction
    p = _greedy_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Phase 2: 2-opt local search
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        p, cost = _local_search_2opt(a, b, p, max_time=remaining_time - 5.0)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    # Phase 3: Random restarts if time allows
    remaining_time = deadline - time.time()
    rng = np.random.default_rng(42)
    while remaining_time > 10.0:
        # Random permutation
        p_rand = rng.permutation(n)
        cost_rand = _qap_cost(a, b, p_rand)
        if cost_rand < best_cost:
            best_cost = cost_rand
            best_p = p_rand.copy()
        
        # Local search on this random start
        time_for_ls = min(remaining_time - 5.0, 15.0)
        if time_for_ls > 2.0:
            p_ls, cost_ls = _local_search_2opt(a, b, p_rand, max_time=time_for_ls)
            if cost_ls < best_cost:
                best_cost = cost_ls
                best_p = p_ls.copy()
        
        remaining_time = deadline - time.time()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
