# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = 118.0  # safety margin before 120s/150s kill

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    # Spectral initialization
    A_sym = (a + a.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    idx_a = np.argmax(eigvals)
    v_a = eigvecs[:, idx_a]

    B_sym = (b + b.T) / 2.0
    eigvals_b, eigvecs_b = np.linalg.eigh(B_sym)
    idx_b = np.argmax(eigvals_b)
    v_b = eigvecs_b[:, idx_b]

    fac_order = np.argsort(-v_a, kind="stable")
    loc_order = np.argsort(v_b, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[fac_order] = loc_order

    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central

    cost_spectral = compute_cost(p)
    cost_rank = compute_cost(p_rank)
    if cost_rank < cost_spectral:
        p = p_rank.copy()
        best_cost = cost_rank
    else:
        best_cost = cost_spectral

    # Vectorized local search
    # Precompute a_row = a[i,:] and a_col = a[:,i]
    # For swap (i,j) with r=p[i], s=p[j]:
    # delta = a[i,:] @ (b[s,p]-b[r,p]) + a[:,i] @ (b[p,s]-b[p,r])
    #         + a[j,:] @ (b[r,p]-b[s,p]) + a[:,j] @ (b[p,r]-b[p,s])
    #         - a[i,j]*(b[s,r]-b[r,s]) - a[j,i]*(b[r,s]-b[s,r])
    #         + a[i,i]*(b[s,s]-b[r,r]) + a[j,j]*(b[r,r]-b[s,s])
    #         + a[i,i]*(b[s,s]-b[r,r]) + a[j,j]*(b[r,r]-b[s,s])
    # Wait, let me re-derive carefully.
    # C = sum_{i,j} a[i,j] b[p[i], p[j]]
    # Swap p[i] <-> p[j]. Let r=p[i], s=p[j].
    # Terms involving p[i] as first index of b: sum_k a[i,k] b[p[i], p[k]] -> sum_k a[i,k] b[s, p'[k]]
    # Terms involving p[i] as second index of b: sum_k a[k,i] b[p[k], p[i]] -> sum_k a[k,i] b[p'[k], s]
    # Similarly for j.
    #
    # delta = sum_k a[i,k](b[s,p'[k]]-b[r,p[k]]) + sum_k a[k,i](b[p'[k],s]-b[p[k],r])
    #         + sum_k a[j,k](b[r,p'[k]]-b[s,p[k]]) + sum_k a[k,j](b[p'[k],r]-b[p[k],s])
    #
    # For k not in {i,j}: p'[k]=p[k]
    # For k=i: p'[i]=s, p[i]=r
    # For k=j: p'[j]=r, p[j]=s
    #
    # Let's split:
    # Sum1 = sum_k a[i,k](b[s,p'[k]]-b[r,p[k]])
    #   = sum_{k!=i,j} a[i,k](b[s,p[k]]-b[r,p[k]]) + a[i,i](b[s,s]-b[r,r]) + a[i,j](b[s,r]-b[r,s])
    # Sum2 = sum_k a[k,i](b[p'[k],s]-b[p[k],r])
    #   = sum_{k!=i,j} a[k,i](b[p[k],s]-b[p[k],r]) + a[i,i](b[s,s]-b[r,r]) + a[j,i](b[r,s]-b[s,r])
    # Sum3 = sum_k a[j,k](b[r,p'[k]]-b[s,p[k]])
    #   = sum_{k!=i,j} a[j,k](b[r,p[k]]-b[s,p[k]]) + a[j,i](b[r,s]-b[s,r]) + a[j,j](b[r,r]-b[s,s])
    # Sum4 = sum_k a[k,j](b[p'[k],r]-b[p[k],s])
    #   = sum_{k!=i,j} a[k,j](b[p[k],r]-b[p[k],s]) + a[i,j](b[s,r]-b[r,s]) + a[j,j](b[r,r]-b[s,s])
    #
    # delta = Sum1+Sum2+Sum3+Sum4
    #
    # Vectorized:
    # Let diff_row = b[s,:] - b[r,:]  (shape n)
    # Let diff_col = b[:,s] - b[:,r]  (shape n)
    # Then for k not in {i,j}:
    #   contribution = a[i,k]*diff_row[p[k]] + a[k,i]*diff_col[p[k]] + a[j,k]*(-diff_row[p[k]]) + a[k,j]*(-diff_col[p[k]])
    #                = (a[i,k]-a[j,k])*diff_row[p[k]] + (a[k,i]-a[k,j])*diff_col[p[k]]
    #
    # So:
    # base = (a[i,:]-a[j,:]) @ diff_row[p] + (a[:,i]-a[:,j]) @ diff_col[p]
    # But this includes k=i and k=j terms incorrectly.
    # For k=i: base includes (a[i,i]-a[j,i])*diff_row[p[i]] + (a[i,i]-a[i,j])*diff_col[p[i]]
    #   = (a[i,i]-a[j,i])*diff_row[r] + (a[i,i]-a[i,j])*diff_col[r]
    # But we need:
    #   from Sum1: a[i,i](b[s,s]-b[r,r]) = a[i,i]*diff_row[s]  -- wait diff_row[s] = b[s,s]-b[r,s]
    #   Hmm, diff_row[p[k]] for k=i is diff_row[r] = b[s,r]-b[r,r]. But we need b[s,s]-b[r,r].
    #   So the vectorized form doesn't directly work because p'[k] differs from p[k] for k in {i,j}.
    #
    # Let me just compute it as:
    # delta = (a[i,:]-a[j,:]) @ (b[s,p]-b[r,p]) + (a[:,i]-a[:,j]) @ (b[p,s]-b[p,r])
    #         + correction_terms
    #
    # Actually, let me just do it the straightforward O(n) way but vectorize over pairs.
    # For n=100, checking all 4950 pairs with O(n) work is 4950*100 = 495000 ops per pass.
    # In numpy, I can vectorize this.
    
    def local_search(p, time_limit):
        n = len(p)
        p = p.copy()
        cost = compute_cost(p)
        improved = True
        while improved:
            if time.time() - t_start > time_limit:
                break
            improved = False
            # Vectorized delta computation for all pairs
            # For each pair (i,j), r=p[i], s=p[j]
            # delta = (a[i,:]-a[j,:]) @ (b[s,p]-b[r,p]) + (a[:,i]-a[:,j]) @ (b[p,s]-b[p,r])
            #         + a[i,i]*(b[s,s]-b[r,s]-diff_row[r]) ... this is getting complicated.
            #
            # Let me just do it pair by pair but with numpy operations.
            for i in range(n):
                if time.time() - t_start > time_limit:
                    break
                for j in range(i+1, n):
                    r, s = p[i], p[j]
                    if r == s:
                        continue
                    # Compute delta
                    # Using the derived formula:
                    # delta = sum_{k not i,j} [(a[i,k]-a[j,k])*diff_row[p[k]] + (a[k,i]-a[k,j])*diff_col[p[k]]]
                    #         + a[i,i]*(b[s,s]-b[r,r]) + a[i,j]*(b[s,r]-b[r,s])
                    #         + a[i,i]*(b[s,s]-b[r,r]) + a[j,i]*(b[r,s]-b[s,r])
                    #         + a[j,i]*(b[r,s]-b[s,r]) + a[j,j]*(b[r,r]-b[s,s])
                    #         + a[i,j]*(b[s,r]-b[r,s]) + a[j,j]*(b[r,r]-b[s,s])
                    #
                    # Let me simplify:
                    # diff_row = b[s,:] - b[r,:]
                    # diff_col = b[:,s] - b[:,r]
                    # base = (a[i,:]-a[j,:]) @ diff_row[p] + (a[:,i]-a[:,j]) @ diff_col[p]
                    # This includes k=i and k=j terms:
                    # k=i: (a[i,i]-a[j,i])*diff_row[r] + (a[i,i]-a[i,j])*diff_col[r]
                    #      = (a[i,i]-a[j,i])*(b[s,r]-b[r,r]) + (a[i,i]-a[i,j])*(b[r,s]-b[r,r])
                    # k=j: (a[i,j]-a[j,j])*diff_row[s] + (a[j,i]-a[j,j])*diff_col[s]
                    #      = (a[i,j]-a[j,j])*(b[s,s]-b[r,s]) + (a[j,i]-a[j,j])*(b[s,s]-b[r,s])
                    #
                    # But we need:
                    # k=i: a[i,i]*(b[s,s]-b[r,r]) + a[i,j]*(b[s,r]-b[r,s]) + a[i,i]*(b[s,s]-b[r,r]) + a[j,i]*(b[r,s]-b[s,r])
                    #      = 2*a[i,i]*(b[s,s]-b[r,r]) + (a[i,j]-a[j,i])*(b[s,r]-b[r,s])
                    # k=j: a[j,i]*(b[r,s]-b[s,r]) + a[j,j]*(b[r,r]-b[s,s]) + a[i,j]*(b[s,r]-b[r,s]) + a[j,j]*(b[r,r]-b[s,s])
                    #      = 2*a[j,j]*(b[r,r]-b[s,s]) + (a[i,j]-a[j,i])*(b[s,r]-b[r,s])
                    #
                    # So correction = needed - base_included
                    # This is getting too complex. Let me just compute delta directly.
                    
                    # Direct computation:
                    # old_i = sum_k a[i,k]*b[r,p[k]] + sum_k a[k,i]*b[p[k],r]
                    # new_i = sum_k a[i,k]*b[s,p_new[k]] + sum_k a[k,i]*b[p_new[k],s]
                    # where p_new[i]=s, p_new[j]=r
                    #
                    # For efficiency, let me compute:
                    # old_i_row = a[i,:] @ b[r,p]  -- but p[k] for k=i is r, for k=j is s
                    # This is just a[i,:] @ b[r,p] which is correct for the old state.
                    # new_i_row: a[i,k]*b[s,p_new[k]] = a[i,i]*b[s,s] + a[i,j]*b[s,r] + sum_{k!=i,j} a[i,k]*b[s,p[k]]
                    #           = a[i,i]*b[s,s] + a[i,j]*b[s,r] + a[i,:] @ b[s,p] - a[i,i]*b[s,p[i]] - a[i,j]*b[s,p[j]]
                    #           = a[i,i]*b[s,s] + a[i,j]*b[s,r] + a[i,:] @ b[s,p] - a[i,i]*b[s,r] - a[i,j]*b[s,s]
                    #           = a[i,i]*(b[s,s]-b[s,r]) + a[i,j]*(b[s,r]-b[s,s]) + a[i,:] @ b[s,p]
                    #           = a[i,:] @ b[s,p] - (a[i,i]-a[i,j])*(b[s,r]-b[s,s])
                    #
                    # This is getting messy. Let me just compute it directly in O(n).
                    old_i = np.dot(a[i,:], b[r, p]) + np.dot(a[:,i], b[p, r])
                    old_j = np.dot(a[j,:], b[s, p]) + np.dot(a[:,j], b[p, s])
                    # Subtract double-counted terms (i,j) and (j,i)
                    old_i -= a[i,j]*b[r,s] + a[j,i]*b[s,r]
                    old_j -= a[j,i]*b[s,r] + a[i,j]*b[r,s]
                    
                    # New values
                    # p_new[i]=s, p_new[j]=r
                    # new_i = sum_k a[i,k]*b[s,p_new[k]] + sum_k a[k,i]*b[p_new[k],s]
                    # For k not i,j: p_new[k]=p[k]
                    # For k=i: p_new[i]=s
                    # For k=j: p_new[j]=r
                    new_i_row = np.dot(a[i,:], b[s, p]) + a[i,i]*(b[s,s]-b[s,r]) + a[i,j]*(b[s,r]-b[s,s])
                    # Wait, b[s,p] has b[s,p[i]]=b[s,r] and b[s,p[j]]=b[s,s].
                    # So a[i,i]*b[s,p[i]] = a[i,i]*b[s,r] but we need a[i,i]*b[s,s].
                    # a[i,j]*b[s,p[j]] = a[i,j]*b[s,s] but we need a[i,j]*b[s,r].
                    # So new_i_row = a[i,:] @ b[s,p] - a[i,i]*b[s,r] - a[i,j]*b[s,s] + a[i,i]*b[s,s] + a[i,j]*b[s,r]
                    new_i_row = np.dot(a[i,:], b[s, p]) + a[i,i]*(b[s,s]-b[s,r]) + a[i,j]*(b[s,r]-b[s,s])
                    new_i_col = np.dot(a[:,i], b[p, s]) + a[i,i]*(b[s,s]-b[r,s]) + a[j,i]*(b[r,s]-b[s,s])
                    # b[p,s] has b[p[i],s]=b[r,s] and b[p[j],s]=b[s,s].
                    # a[i,i]*b[p[i],s] = a[i,i]*b[r,s] but we need a[i,i]*b[s,s].
                    # a[j,i]*b[p[j],s] = a[j,i]*b[s,s] but we need a[j,i]*b[r,s].
                    # So new_i_col = a[:,i] @ b[p,s] - a[i,i]*b[r,s] - a[j,i]*b[s,s] + a[i,i]*b[s,s] + a[j,i]*b[r,s]
                    new_i_col = np.dot(a[:,i], b[p, s]) + a[i,i]*(b[s,s]-b[r,s]) + a[j,i]*(b[r,s]-b[s,s])
                    new_i = new_i_row + new_i_col - a[i,j]*b[s,r] - a[j,i]*b[r,s]
                    
                    new_j_row = np.dot(a[j,:], b[r, p]) + a[j,i]*(b[r,s]-b[r,r]) + a[j,j]*(b[r,r]-b[r,s])
                    new_j_col = np.dot(a[:,j], b[p, r]) + a[i,j]*(b[s,r]-b[r,r]) + a[j,j]*(b[r,r]-b[s,r])
                    new_j = new_j_row + new_j_col - a[i,j]*b[s,r] - a[j,i]*b[r,s]
                    
                    delta = (new_i + new_j) - (old_i + old_j)
                    
                    if delta < -1e-9:
                        p[i], p[j] = s, r
                        cost += int(delta)
                        improved = True
        
        return p, cost

    # Run local search
    p, best_cost = local_search(p, deadline)

    # Iterated Local Search with LAP-guided perturbation
    rng = np.random.RandomState(42)
    no_improve_count = 0
    max_no_improve = 30

    while time.time() - t_start < deadline - 3.0:
        # LAP-guided perturbation: solve a linear assignment problem
        # using the current solution as a reference
        # C_ij = a[i,:] @ b[p, :].T ... no, let's use a simpler surrogate.
        # Surrogate: min sum_i (sum_j a[i,j]) * b[p[i], c_i] where c_i is some reference location.
        # Or: min sum_i sum_j a[i,j] * b[p[i], p[j]] approximated by
        # min sum_i (a[i,:] + a[:,i]) @ b[p[i], :]  (ignoring the p[j] dependence)
        # Actually, a common approach: use the reduced cost from the current solution.
        
        # Simpler: random 2-5 swaps
        n_swaps = rng.randint(2, 6)
        p_pert = p.copy()
        for _ in range(n_swaps):
            i, j = rng.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        
        p_new, cost_new = local_search(p_pert, deadline)
        
        if cost_new < best_cost:
            p = p_new
            best_cost = cost_new
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
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
