# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = 110.0  # safety margin before 120s kill

    # Precompute symmetric parts (QAP cost is sum a[i,j]*b[p[i],p[j]], 
    # we can work with the full matrices directly)
    # For delta computation of swapping p[i] and p[j]:
    # delta = sum_k a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]]) + a[k,i]*(b[p[j],p[k]] - b[p[i],p[k]])
    #        + sum_k a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]]) + a[k,j]*(b[p[i],p[k]] - b[p[j],p[k]])
    #        - a[i,j]*(b[p[i],p[j]] - b[p[j],p[i]]) - a[j,i]*(b[p[j],p[i]] - b[p[i],p[j]])
    # Simplified: delta(i,j) = (a[i,:]+a[:,i])@(b[p[j],:]-b[p[i],:]) 
    #                          + (a[j,:]+a[:,j])@(b[p[i],:]-b[p[j],:])
    #                          - 2*(a[i,j]*b[p[i],p[j]] + a[j,i]*b[p[j],p[i]] 
    #                               - a[i,j]*b[p[j],p[i]] - a[j,i]*b[p[i],p[j]])
    # Actually let me be more careful.
    
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # After swapping positions of facilities i and j (i.e., p[i]<->p[j]):
    # The change involves all terms where p[i] or p[j] appear.
    
    # Let me use a cleaner formulation:
    # C(p) = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # = sum_i (a[i,:] @ b[p[i], :])  [summing over j: a[i,j]*b[p[i],p[j]]]
    # Actually C = sum_{i,j} a[i,j] * b[p[i],p[j]]
    
    # For swap of p[i] <-> p[j]:
    # Only terms involving index i or j in the facility dimension change.
    # C_new - C_old = 
    #   For facility i: sum_k a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]])
    #   For facility j: sum_k a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]])
    #   But we also need to account for k=i and k=j properly since p[i],p[j] swap.
    
    # Let me just compute it directly:
    # Let r = p[i], s = p[j] (the two locations being swapped)
    # Terms that change: those with facility index i or j.
    # For facility i, location changes from r to s:
    #   sum_k a[i,k] * (b[s, p[k]] - b[r, p[k]])
    # For facility j, location changes from s to r:
    #   sum_k a[j,k] * (b[r, p[k]] - b[s, p[k]])
    # But when k=i, p[k] was r and becomes s; when k=j, p[k] was s and becomes r.
    # So we need to be careful. Let me just compute delta directly.
    
    # delta(i,j) where we swap p[i] and p[j]:
    # Let r = p[i], s = p[j]
    # Old contribution involving i: sum_k a[i,k]*b[r, p[k]]
    # New contribution involving i: sum_k a[i,k]*b[s, p_new[k]]
    #   where p_new[i]=s, p_new[j]=r, p_new[k]=p[k] for k!=i,j
    # This is getting complex. Let me use a simpler approach:
    # delta = (a[i,:]+a[:,i]) @ (b[s,:] - b[r,:])  ... no that's not right either.
    
    # Let me just do it the straightforward way:
    # delta(i,j) = sum_k a[i,k]*(b[s,p[k]]-b[r,p[k]]) + sum_k a[k,i]*(b[s,p[k]]-b[r,p[k]])
    #             + sum_k a[j,k]*(b[r,p[k]]-b[s,p[k]]) + sum_k a[k,j]*(b[r,p[k]]-b[s,p[k]])
    #             - a[i,j]*(b[s,r]-b[r,s]) - a[j,i]*(b[r,s]-b[s,r])
    #             + a[i,i]*(b[s,s]-b[r,r]) + a[j,j]*(b[r,r]-b[s,s])  ... 
    # 
    # This is getting messy. Let me use a different approach:
    # Precompute for each location l: B_row[l] = b[l, :]
    # For a given permutation p, the cost contribution of facility i at location p[i] is:
    #   sum_j a[i,j] * b[p[i], p[j]]  (row part)
    #   + sum_j a[j,i] * b[p[j], p[i]]  (column part, but this is already counted in the full sum)
    # Actually the full cost C = sum_{i,j} a[i,j]*b[p[i],p[j]] counts all (i,j) pairs.
    
    # Simplest correct delta for swapping p[i]<->p[j] (let r=p[i], s=p[j]):
    # C = sum_{i,j} a[i,j]*b[p[i],p[j]]
    # After swap, p'[i]=s, p'[j]=r, p'[k]=p[k] else.
    # C' - C = sum_j a[i,j]*(b[s,p'[j]]-b[r,p[j]]) + sum_j a[j,i]*(b[p'[j],s]-b[p[j],r])
    #         + sum_j a[j,k]... no, only i and j rows/cols change.
    # 
    # Let me just enumerate: the terms that change are those where the first index is i or j,
    # OR the second index's location changes (which happens when the facility is i or j).
    # Actually both indices matter. Let me think of it as:
    # C = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # The only things that change are p[i] and p[j]. So:
    # C' - C = sum_j a[i,j]*(b[s,p'[j]] - b[r,p[j]]) 
    #        + sum_j a[j,i]*(b[p'[j],s] - b[p[j],r])
    #        + sum_j a[j,k]*(b[p'[j],p'[k]] - b[p[j],p[k]])  for j,k not i... 
    # No! Only p[i] and p[j] change. So b[p[i],p[j]] changes, b[p[j],p[i]] changes,
    # b[p[i],p[k]] changes for all k, b[p[k],p[i]] changes for all k,
    # b[p[j],p[k]] changes for all k, b[p[k],p[j]] changes for all k.
    # But b[p[i],p[j]] and b[p[j],p[i]] are included in the above.
    
    # delta = sum_{k} a[i,k]*(b[s, p'[k]] - b[r, p[k]])
    #       + sum_{k} a[k,i]*(b[p'[k], s] - b[p[k], r])
    #       + sum_{k} a[j,k]*(b[r, p'[k]] - b[s, p[k]])
    #       + sum_{k} a[k,j]*(b[p'[k], r] - b[p[k], s])
    # where p'[i]=s, p'[j]=r, p'[k]=p[k] for k not in {i,j}
    # 
    # For k not in {i,j}: p'[k]=p[k]
    # For k=i: p'[i]=s, p[i]=r
    # For k=j: p'[j]=r, p[j]=s
    #
    # So:
    # delta = sum_{k not i,j} a[i,k]*(b[s,p[k]]-b[r,p[k]]) + a[i,i]*(b[s,s]-b[r,r]) + a[i,j]*(b[s,r]-b[r,s])
    #       + sum_{k not i,j} a[k,i]*(b[p[k],s]-b[p[k],r]) + a[i,i]*(b[s,s]-b[r,r]) + a[j,i]*(b[r,s]-b[s,r])
    #       + sum_{k not i,j} a[j,k]*(b[r,p[k]]-b[s,p[k]]) + a[j,i]*(b[r,s]-b[s,r]) + a[j,j]*(b[r,r]-b[s,s])
    #       + sum_{k not i,j} a[k,j]*(b[p[k],r]-b[p[k],s]) + a[i,j]*(b[s,r]-b[r,s]) + a[j,j]*(b[r,r]-b[s,s])
    
    # This is complex. For n=100, I can just compute the delta in a vectorized way per pair.
    # Actually, for n=100, a full local search iteration checking all O(n^2)=10000 pairs 
    # with O(n) work each is 10^6 operations, which is fast. But let me vectorize.
    
    # Alternative: for each facility i, precompute row_a_i = a[i,:] and col_a_i = a[:,i]
    # For swap (i,j) with r=p[i], s=p[j]:
    # delta = (a[i,:] + a[:,i]) @ (b[s,:] - b[r,:])  -- NO, this isn't right because
    # the p[k] values matter.
    
    # Let me just do it O(n) per pair, vectorized over pairs using numpy.
    # For n=100, 10000 pairs * 100 ops = 10^6, very fast.
    
    # Actually, let me precompute:
    # For each location l, let B_l = b[l, :] (row) and B_col_l = b[:, l] (column)
    # For a permutation p, define for facility i at location p[i]:
    #   row_contrib[i] = sum_j a[i,j] * b[p[i], p[j]] = a[i,:] @ B_row[p[i]]  where B_row[l]=b[l,:]
    #   Wait, sum_j a[i,j]*b[p[i],p[j]] = a[i,:] dot b[p[i], p[0],p[1],...]
    #   This is a[i,:] @ b[p[i], p]  where p is the vector of locations.
    #   = a[i,:] @ (b[p[i], :][p])  -- no, b[p[i], p[j]] for each j.
    #   Let me define: for location l, B_l = b[l, :] (shape n)
    #   Then sum_j a[i,j]*b[p[i],p[j]] = sum_j a[i,j]*B_{p[i]}[p[j]] = a[i,:] @ B_{p[i]}[p]
    #   where B_{p[i]}[p] means indexing row B_{p[i]} by the array p.
    #   = a[i,:] @ (b[p[i], :])[p]  -- this is a dot product of a[i,:] with the permuted row.
    
    # OK this is getting complicated. Let me just use a straightforward approach:
    # Compute cost, and for local search, compute delta for each pair (i,j) in O(n) time.
    # With n=100, that's 10000*100 = 1e6 per full pass, which is ~1ms in numpy.
    
    # Let me write a clean implementation:
    
    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))
    
    # Spectral initialization
    # Compute eigenvectors of A (flow matrix)
    # The idea: the dominant eigenvector of A captures the main flow structure.
    # We want to align high-flow facilities with locations that have good distance properties.
    
    # Symmetrize A for eigen decomposition
    A_sym = (a + a.T) / 2.0
    # Get eigenvectors
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    # The eigenvector with largest magnitude (most negative eigenvalue for minimization)
    # captures the main structure.
    
    # For the distance matrix B, we want to find an ordering that minimizes
    # the alignment with the flow structure.
    # Use the dominant eigenvector of A to rank facilities.
    # Then assign them to locations ranked by their "centrality" in B.
    
    # Actually, a better spectral approach:
    # The QAP can be approximated by:
    # min sum_{i,j} a[i,j]*b[p[i],p[j]] ≈ min sum_i (a[i,:] * a[:,i])_sum * b[p[i],p[i]] ... no
    # 
    # Simpler spectral init: use the eigenvector of A with largest |eigenvalue|
    # to order facilities, and the eigenvector of B with largest |eigenvalue| to order locations.
    # Then align them (or anti-align for minimization).
    
    # Let me try: rank facilities by their projection onto the dominant eigenvector of A
    # and rank locations by their projection onto the dominant eigenvector of B.
    # For minimization, we want high-flow facilities at low-distance locations.
    
    # Dominant eigenvector of A (largest eigenvalue)
    idx_a = np.argmax(eigvals)
    v_a = eigvecs[:, idx_a]
    
    # For B, compute its dominant eigenvector
    B_sym = (b + b.T) / 2.0
    eigvals_b, eigvecs_b = np.linalg.eigh(B_sym)
    idx_b = np.argmax(eigvals_b)
    v_b = eigvecs_b[:, idx_b]
    
    # Rank facilities by v_a (high value = high flow structure)
    # Rank locations by v_b (high value = high distance structure)
    # For minimization: high flow -> low distance
    # So sort facilities by v_a descending, locations by v_b ascending
    
    fac_order = np.argsort(-v_a, kind="stable")  # high flow first
    loc_order = np.argsort(v_b, kind="stable")    # low distance first
    
    p = np.empty(n, dtype=np.int64)
    p[fac_order] = loc_order
    
    # Also try the anti-alignment (high flow -> high distance is bad, so this should be worse)
    # and the rank-matching as a fallback
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    
    # Start with the better of the two
    cost_spectral = compute_cost(p)
    cost_rank = compute_cost(p_rank)
    if cost_rank < cost_spectral:
        p = p_rank.copy()
        best_cost = cost_rank
    else:
        best_cost = cost_spectral
    
    # Local search: first-improvement 2-opt
    def local_search(p, time_limit):
        """First-improvement local search for QAP."""
        n = len(p)
        p = p.copy()
        cost = compute_cost(p)
        
        # Precompute for faster delta computation
        # For swap (i,j) with r=p[i], s=p[j]:
        # delta = sum_k a[i,k]*(b[s,p[k]]-b[r,p[k]]) + sum_k a[k,i]*(b[p[k],s]-b[p[k],r])
        #         + sum_k a[j,k]*(b[r,p[k]]-b[s,p[k]]) + sum_k a[k,j]*(b[p[k],r]-b[p[k],s])
        #         - a[i,j]*(b[s,r]-b[r,s]) - a[j,i]*(b[r,s]-b[s,r])  (correction for double-counting)
        # Wait, I need to be more careful. Let me just compute it directly.
        
        # Actually, let me use a simpler but correct formula:
        # C = sum_{i,j} a[i,j]*b[p[i],p[j]]
        # After swapping p[i]<->p[j] (r=p[i], s=p[j]):
        # Only terms where p[i] or p[j] appear change.
        # p[i] appears as first index of b in terms a[i,k] (for all k) and as second index in a[k,i] (for all k)
        # Similarly for p[j].
        # 
        # delta = sum_k a[i,k]*(b[s,p'[k]]-b[r,p[k]]) + sum_k a[k,i]*(b[p'[k],s]-b[p[k],r])
        #         + sum_k a[j,k]*(b[r,p'[k]]-b[s,p[k]]) + sum_k a[k,j]*(b[p'[k],r]-b[p[k],s])
        # where p' differs from p only at i and j.
        #
        # For k not in {i,j}: p'[k]=p[k]
        # For k=i: p'[i]=s, p[i]=r  -> b[s,s] vs b[r,r], b[s,s] vs b[r,r]
        # For k=j: p'[j]=r, p[j]=s  -> b[s,r] vs b[r,s], b[r,s] vs b[s,r]
        
        # Let me just compute it as:
        # delta = (a[i,:] @ (b[s,p]-b[r,p])) + (a[:,i] @ (b[p,s]-b[p,r]))
        #         + (a[j,:] @ (b[r,p]-b[s,p])) + (a[:,j] @ (b[p,r]-b[p,s]))
        # where b[s,p] means the vector [b[s,p[0]], b[s,p[1]], ...]
        # and b[p,s] means [b[p[0],s], b[p[1],s], ...]
        # But this double-counts the i,j terms. Let me just compute directly for correctness.
        
        improved = True
        while improved:
            if time.time() - t_start > time_limit:
                break
            improved = False
            for i in range(n):
                if time.time() - t_start > time_limit:
                    break
                for j in range(i+1, n):
                    r, s = p[i], p[j]
                    if r == s:
                        continue
                    # Compute delta for swapping p[i] and p[j]
                    # b[r,p] = b[r, p[0:n]] as a vector
                    # b[s,p] = b[s, p[0:n]]
                    # b[p,r] = b[p[0:n], r] as a vector
                    # b[p,s] = b[p[0:n], s]
                    br_p = b[r, p]  # shape (n,)
                    bs_p = b[s, p]
                    bp_r = b[p, r]
                    bp_s = b[p, s]
                    
                    # delta = a[i,:] @ (bs_p - br_p) + a[:,i] @ (bp_s - bp_r)
                    #         + a[j,:] @ (br_p - bs_p) + a[:,j] @ (bp_r - bp_s)
                    #         - correction for (i,j) and (j,i) terms being double-counted
                    # 
                    # Actually, the formula above is correct if we think of it as:
                    # The change in C due to p[i] changing from r to s:
                    #   sum_k a[i,k]*(b[s,p[k]]-b[r,p[k]])  [first index is i]
                    #   sum_k a[k,i]*(b[p[k],s]-b[p[k],r])  [second index is i]
                    # The change in C due to p[j] changing from s to r:
                    #   sum_k a[j,k]*(b[r,p[k]]-b[s,p[k]])  [first index is j]
                    #   sum_k a[k,j]*(b[p[k],r]-b[p[k],s])  [second index is j]
                    #
                    # But when k=i in the first sum, p[k]=p[i]=r (old), but after swap p'[i]=s.
                    # So b[s,p'[i]]=b[s,s] not b[s,p[i]]=b[s,r].
                    # Similarly for other cross terms.
                    #
                    # The formula a[i,:] @ (bs_p - br_p) uses p[k] for all k, but for k=i,
                    # the new p[i]=s, not r. So we need to correct.
                    #
                    # Let me just compute it properly:
                    # delta = sum_k a[i,k]*(b[s,p_new[k]]-b[r,p[k]])
                    #       + sum_k a[k,i]*(b[p_new[k],s]-b[p[k],r])
                    #       + sum_k a[j,k]*(b[r,p_new[k]]-b[s,p[k]])
                    #       + sum_k a[k,j]*(b[p_new[k],r]-b[p[k],s])
                    # where p_new[i]=s, p_new[j]=r, p_new[k]=p[k] else.
                    
                    # For k not in {i,j}: p_new[k]=p[k], so terms are:
                    #   a[i,k]*(b[s,p[k]]-b[r,p[k]]) + a[k,i]*(b[p[k],s]-b[p[k],r])
                    #   + a[j,k]*(b[r,p[k]]-b[s,p[k]]) + a[k,j]*(b[p[k],r]-b[p[k],s])
                    # For k=i: p_new[i]=s, p[i]=r
                    #   a[i,i]*(b[s,s]-b[r,r]) + a[i,i]*(b[s,s]-b[r,r])
                    #   + a[j,i]*(b[r,s]-b[s,s]) + a[i,j]*(b[s,r]-b[r,s])  -- wait
                    #   a[j,i]*(b[r,p_new[i]]-b[s,p[i]]) = a[j,i]*(b[r,s]-b[s,r])
                    #   a[i,j]*(b[p_new[i],r]-b[p[i],s]) = a[i,j]*(b[s,r]-b[r,s])
                    # For k=j: p_new[j]=r, p[j]=s
                    #   a[i,j]*(b[s,r]-b[r,s]) + a[j,i]*(b[r,s]-b[s,r])
                    #   + a[j,j]*(b[r,r]-b[s,s]) + a[j,j]*(b[r,r]-b[s,s])
                    
                    # This is getting very messy. Let me just compute the cost difference
                    # directly by computing the relevant terms.
                    
                    # Simpler: just compute C_old and C_new for the affected terms.
                    # Affected terms: all (i,k), (k,i), (j,k), (k,j) for all k.
                    # But (i,j) and (j,i) are in both sets.
                    
                    # Let me just do:
                    # old = sum_k a[i,k]*b[r,p[k]] + sum_k a[k,i]*b[p[k],r]
                    #       + sum_k a[j,k]*b[s,p[k]] + sum_k a[k,j]*b[p[k],s]
                    #       - a[i,j]*b[r,s] - a[j,i]*b[s,r]  (subtract double-counted)
                    # new = sum_k a[i,k]*b[s,p_new[k]] + sum_k a[k,i]*b[p_new[k],s]
                    #       + sum_k a[j,k]*b[r,p_new[k]] + sum_k a[k,j]*b[p_new[k],r]
                    #       - a[i,j]*b[s,r] - a[j,i]*b[r,s]
                    
                    # Where p_new[i]=s, p_new[j]=r, p_new[k]=p[k] else.
                    
                    # For the sums, split into k in {i,j} and k not in {i,j}:
                    # For k not in {i,j}: p_new[k]=p[k]
                    #   old_part = sum_{k not i,j} [a[i,k]*b[r,p[k]] + a[k,i]*b[p[k],r] + a[j,k]*b[s,p[k]] + a[k,j]*b[p[k],s]]
                    #   new_part = sum_{k not i,j} [a[i,k]*b[s,p[k]] + a[k,i]*b[p[k],s] + a[j,k]*b[r,p[k]] + a[k,j]*b[p[k],r]]
                    # For k=i:
                    #   old: a[i,i]*b[r,r] + a[i,i]*b[r,r] + a[j,i]*b[s,r] + a[i,j]*b[r,s]
                    #        Wait, a[j,i]*b[s,p[i]]=a[j,i]*b[s,r] and a[i,j]*b[p[i],s]=a[i,j]*b[r,s]
                    #   new: a[i,i]*b[s,s] + a[i,i]*b[s,s] + a[j,i]*b[r,s] + a[i,j]*b[s,r]
                    # For k=j:
                    #   old: a[i,j]*b[r,s] + a[j,i]*b[s,r] + a[j,j]*b[s,s] + a[j,j]*b[s,s]
                    #        a[i,j]*b[r,p[j]]=a[i,j]*b[r,s], a[j,i]*b[p[j],r]=a[j,i]*b[s,r]
                    #        a[j,j]*b[s,p[j]]=a[j,j]*b[s,s], a[j,j]*b[p[j],s]=a[j,j]*b[s,s]
                    #   new: a[i,j]*b[s,r] + a[j,i]*b[r,s] + a[j,j]*b[r,r] + a[j,j]*b[r,r]
                    
                    # delta = new - old
                    # This is O(n) per pair. Let me just compute it.
                    
                    # For efficiency, precompute:
                    # br = b[r, :], bs = b[s, :], col_r = b[:, r], col_s = b[:, s]
                    # For k not in {i,j}:
                    #   term = a[i,k]*(bs[p[k]]-br[p[k]]) + a[k,i]*(col_s[p[k]]-col_r[p[k]])
                    #         + a[j,k]*(br[p[k]]-bs[p[k]]) + a[k,j]*(col_r[p[k]]-col_s[p[k]])
                    #         = (a[i,k]-a[j,k])*(bs[p[k]]-br[p[k]]) + (a[k,i]-a[k,j])*(col_s[p[k]]-col_r[p[k]])
                    
                    # Let me just compute it directly, it's O(n) and n=100.
                    delta = 0.0
                    # k not in {i,j}
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        pk = p[k]
                        delta += a[i,k]*(b[s,pk]-b[r,pk]) + a[k,i]*(b[pk,s]-b[pk,r])
                        delta += a[j,k]*(b[r,pk]-b[s,pk]) + a[k,j]*(b[pk,r]-b[pk,s])
                    # k=i
                    delta += a[i,i]*(b[s,s]-b[r,r]) + a[i,i]*(b[s,s]-b[r,r])
                    delta += a[j,i]*(b[r,s]-b[s,r]) + a[i,j]*(b[s,r]-b[r,s])
                    # k=j
                    delta += a[i,j]*(b[s,r]-b[r,s]) + a[j,i]*(b[r,s]-b[s,r])
                    delta += a[j,j]*(b[r,r]-b[s,s]) + a[j,j]*(b[r,r]-b[s,s])
                    
                    if delta < -1e-9:
                        p[i], p[j] = s, r
                        cost += int(delta)
                        improved = True
                        # Continue checking other pairs (first-improvement)
        
        return p, cost
    
    # Run local search
    p, best_cost = local_search(p, deadline)
    
    # Iterated Local Search: perturb and re-optimize
    rng = np.random.RandomState(42)
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - t_start < deadline - 5.0:
        # Perturb: 2-5 random swaps
        n_swaps = rng.randint(2, 6)
        p_pert = p.copy()
        for _ in range(n_swaps):
            i, j = rng.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        
        # Local search from perturbed
        p_new, cost_new = local_search(p_pert, deadline)
        
        if cost_new < best_cost:
            p = p_new
            best_cost = cost_new
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Reset and try a different perturbation strategy
                no_improve_count = 0
                n_swaps = rng.randint(5, 11)
                p_pert = p.copy()
                for _ in range(n_swaps):
                    i, j = rng.choice(n, 2, replace=False)
                    p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
                p_new, cost_new = local_search(p_pert, deadline)
                if cost_new < best_cost:
                    p = p_new
                    best_cost = cost_new
    
    return p.tolist()
# EVOLVE-BLOCK-END
