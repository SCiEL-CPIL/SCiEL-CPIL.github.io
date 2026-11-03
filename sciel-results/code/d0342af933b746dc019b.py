"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    deadline = time.time() + 110.0  # safety margin before 120s kill

    # --- 1. Constructive heuristic: rank-matching initial solution ---
    # Heavy flow rows to central (low-distance) location rows
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by decreasing flow sum, locations by increasing distance sum
    heavy_order = np.argsort(-a_row_sums, kind='stable')
    central_order = np.argsort(b_row_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    # --- 2. Compute initial cost ---
    def qap_cost(perm):
        # cost = sum_{i,j} a[i,j] * b[perm[i], perm[j]]
        # = trace(a @ b[perm][:, perm])  -- but we need the full sum
        # Actually: sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Let Bp = b[perm][:, perm]  (i.e., b with rows and cols permuted)
        # Then cost = sum_{i,j} a[i,j] * Bp[i,j] = np.sum(a * Bp)
        Bp = b[perm][:, perm]
        return int(np.sum(a.astype(np.int64) * Bp))
    
    best_cost = qap_cost(p)
    best_p = p.copy()
    
    # --- 3. Local search with 2-opt swaps (correct delta computation) ---
    # For swapping locations of facilities i and j:
    # delta = sum_k a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]]) + a[k,i]*(b[p[j],p[k]] - b[p[i],p[k]])
    #       + sum_k a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]]) + a[k,j]*(b[p[i],p[k]] - b[p[j],p[k]])
    #       - a[i,j]*(b[p[i],p[j]] - b[p[j],p[i]])  -- but b is symmetric so this is 0
    #       - a[j,i]*(b[p[j],p[i]] - b[p[i],p[j]])  -- also 0 if symmetric
    #
    # More carefully, let's derive:
    # Original contribution involving i or j:
    #   sum_k a[i,k]*b[p[i],p[k]] + sum_k a[k,i]*b[p[k],p[i]]  (terms where i is first or second)
    #   sum_k a[j,k]*b[p[j],p[k]] + sum_k a[k,j]*b[p[k],p[j]]
    #   - a[i,j]*b[p[i],p[j]] - a[j,i]*b[p[j],p[i]]  (double-counted cross terms)
    #
    # After swap (p[i]<->p[j]):
    #   sum_k a[i,k]*b[p[j],p[k]] + sum_k a[k,i]*b[p[k],p[j]]
    #   sum_k a[j,k]*b[p[i],p[k]] + sum_k a[k,j]*b[p[k],p[i]]
    #   - a[i,j]*b[p[j],p[i]] - a[j,i]*b[p[i],p[j]]
    #
    # delta = new - old
    #
    # We can precompute for each facility i and location l:
    #   row_contrib[i, l] = sum_k a[i,k]*b[l, p[k]]  (i placed at l, others at p[k])
    #   col_contrib[i, l] = sum_k a[k,i]*b[p[k], l]  (i placed at l, others at p[k])
    #
    # Then contribution of facility i at location p[i]:
    #   c_i = row_contrib[i, p[i]] + col_contrib[i, p[i]] - a[i,i]*b[p[i],p[i]]  -- wait, this is getting complex
    #
    # Let me use a simpler O(n) delta per swap:
    # delta(i,j) = sum_k [a[i,k] + a[k,i]] * (b[p[j],p[k]] - b[p[i],p[k]])
    #            + sum_k [a[j,k] + a[k,j]] * (b[p[i],p[k]] - b[p[j],p[k]])
    #            - [a[i,j] + a[j,i]] * (b[p[i],p[j]] - b[p[j],p[i]])
    #
    # But b is symmetric (distance matrix), so b[p[i],p[j]] = b[p[j],p[i]], so the last term is 0.
    #
    # So: delta(i,j) = sum_k (a[i,k] + a[k,i]) * (b[p[j],p[k]] - b[p[i],p[k]])
    #                   - sum_k (a[j,k] + a[k,j]) * (b[p[j],p[k]] - b[p[i],p[k]])
    #                 = sum_k [(a[i,k] + a[k,i]) - (a[j,k] + a[k,j])] * (b[p[j],p[k]] - b[p[i],p[k]])
    
    a_sym = a + a.T  # a[i,k] + a[k,i]
    
    # Precompute: for each pair of locations (u, v), the vector diff = b[v, :] - b[u, :]
    # We'll compute delta on the fly using precomputed a_sym rows.
    
    # For efficiency, precompute a_sym as int64
    a_sym = a_sym.astype(np.int64)
    b_int = b.astype(np.int64)
    
    # Local search: iterate over all pairs, accept improving swaps
    # Use a list of positions for quick lookup
    pos = np.empty(n, dtype=np.int64)  # pos[loc] = facility at loc
    pos[p] = np.arange(n)
    
    improved = True
    max_passes = 50
    pass_count = 0
    
    while improved and pass_count < max_passes and time.time() < deadline:
        improved = False
        pass_count += 1
        
        # Shuffle order to avoid bias
        if pass_count > 1:
            idx = np.arange(n)
            np.random.shuffle(idx)
        else:
            idx = np.arange(n)
        
        for ii in range(n):
            if time.time() > deadline - 5:
                break
            i = idx[ii]
            li = p[i]
            
            for jj in range(ii + 1, n):
                if time.time() > deadline - 5:
                    break
                j = idx[jj]
                lj = p[j]
                
                # Compute delta for swapping i and j
                # delta = sum_k [(a_sym[i,k]) - (a_sym[j,k])] * (b[lj, p[k]] - b[li, p[k]])
                diff_a = a_sym[i] - a_sym[j]
                diff_b = b_int[lj, p] - b_int[li, p]
                delta = int(np.dot(diff_a, diff_b))
                
                if delta < 0:
                    # Accept swap
                    p[i], p[j] = p[j], p[i]
                    pos[li], pos[lj] = pos[lj], pos[li]
                    best_cost += delta
                    improved = True
    
    # --- 4. Second phase: random restarts with perturbation + local search ---
    # If we have time, try a few random perturbations of the best solution
    if time.time() < deadline - 30:
        for restart in range(3):
            if time.time() > deadline - 15:
                break
            
            # Perturb: swap 5 random pairs
            p_try = best_p.copy()
            pos_try = np.empty(n, dtype=np.int64)
            pos_try[p_try] = np.arange(n)
            
            for _ in range(5):
                i1, j1 = np.random.choice(n, 2, replace=False)
                l1, l2 = p_try[i1], p_try[j1]
                p_try[i1], p_try[j1] = p_try[j1], p_try[i1]
                pos_try[l1], pos_try[l2] = pos_try[l2], pos_try[l1]
            
            cost_try = qap_cost(p_try)
            
            if cost_try < best_cost:
                # Run local search from this perturbed solution
                p = p_try.copy()
                pos = pos_try.copy()
                cur_cost = cost_try
                
                improved2 = True
                while improved2 and time.time() < deadline - 10:
                    improved2 = False
                    idx = np.arange(n)
                    np.random.shuffle(idx)
                    
                    for ii in range(n):
                        if time.time() > deadline - 10:
                            break
                        i = idx[ii]
                        li = p[i]
                        
                        for jj in range(ii + 1, n):
                            if time.time() > deadline - 10:
                                break
                            j = idx[jj]
                            lj = p[j]
                            
                            diff_a = a_sym[i] - a_sym[j]
                            diff_b = b_int[lj, p] - b_int[li, p]
                            delta = int(np.dot(diff_a, diff_b))
                            
                            if delta < 0:
                                p[i], p[j] = p[j], p[i]
                                pos[li], pos[lj] = pos[lj], pos[li]
                                cur_cost += delta
                                improved2 = True
                    
                    if not improved2:
                        break
                
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_p = p.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
