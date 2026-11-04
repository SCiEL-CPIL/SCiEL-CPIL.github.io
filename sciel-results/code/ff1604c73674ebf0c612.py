import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Helper to compute cost
    def compute_cost(p):
        # p is permutation of locations for facilities 0..n-1
        # Cost = sum a[i,j] * b[p[i], p[j]]
        # Vectorized: (a * b[np.ix_(p, p)]).sum()
        return np.sum(a * b[np.ix_(p, p)])
    
    # Initial solution: Rank matching
    # Heavy flows (high row sum in A) to central locations (low row sum in B)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_sums, kind="stable")
    central = np.argsort(b_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    best_p = p.copy()
    best_cost = compute_cost(best_p)
    
    # Local Search Loop
    # M[i, l] = sum_k a[i, k] * b[l, p[k]]
    # This is A @ B_perm.T where B_perm[l, k] = b[l, p[k]]
    
    while time.time() < deadline:
        # Compute M
        # B_perm is b with columns permuted by p
        # B_perm[l, k] = b[l, p[k]]
        # So M[i, l] = sum_k a[i, k] * b[l, p[k]]
        # M = a @ b[:, p].T ?
        # b[:, p] has shape (n, n) where col k is b[:, p[k]].
        # So (b[:, p]).T has row l as b[:, p][l, :] = b[l, p].
        # So M[i, l] = sum_k a[i, k] * b[l, p[k]]. Correct.
        M = a @ b[:, p].T
        
        # Calculate Delta for all pairs (i, j)
        # Delta(i, j) = M[i, p[j]] + M[j, p[i]] - M[i, p[i]] - M[j, p[j]] 
        # Wait, the formula derived earlier:
        # Delta = M'[i, v] + M'[j, u] - M[i, u] - M[j, v]
        # Where u=p[i], v=p[j]
        # M'[i, v] = M[i, v] - a[i,i]b[v,u] - a[i,j]b[v,v] + a[i,i]b[v,v] + a[i,j]b[v,u]
        # This correction is needed because M[i, v] includes terms with k=i and k=j based on OLD p.
        # Let's re-verify the O(1) delta formula.
        
        # Old Cost terms for i, j:
        # C_old_part = M[i, u] + M[j, v]
        # M[i, u] = sum_k a[i,k] b[u, p[k]] = a[i,i]b[u,u] + a[i,j]b[u,v] + sum_{k!=i,j} a[i,k]b[u,p[k]]
        # M[j, v] = sum_k a[j,k] b[v, p[k]] = a[j,j]b[v,v] + a[j,i]b[v,u] + sum_{k!=i,j} a[j,k]b[v,p[k]]
        
        # New Cost terms for i, j (swapped to v, u):
        # C_new_part = M'[i, v] + M'[j, u]
        # M'[i, v] = sum_k a[i,k] b[v, p'[k]]
        # p'[i]=v, p'[j]=u, p'[k]=p[k] for k!=i,j
        # M'[i, v] = a[i,i]b[v,v] + a[i,j]b[v,u] + sum_{k!=i,j} a[i,k]b[v,p[k]]
        # M'[j, u] = a[j,j]b[u,u] + a[j,i]b[u,v] + sum_{k!=i,j} a[j,k]b[u,p[k]]
        
        # We want to express sum_{k!=i,j} a[i,k]b[v,p[k]] in terms of M.
        # M[i, v] = sum_k a[i,k] b[v, p[k]] = a[i,i]b[v,u] + a[i,j]b[v,v] + sum_{k!=i,j} a[i,k]b[v,p[k]]
        # So sum_{k!=i,j} a[i,k]b[v,p[k]] = M[i, v] - a[i,i]b[v,u] - a[i,j]b[v,v]
        
        # Thus:
        # M'[i, v] = a[i,i]b[v,v] + a[i,j]b[v,u] + (M[i, v] - a[i,i]b[v,u] - a[i,j]b[v,v])
        #          = M[i, v] + a[i,i](b[v,v] - b[v,u]) + a[i,j](b[v,u] - b[v,v])
        #          = M[i, v] + (a[i,i] - a[i,j])(b[v,v] - b[v,u])
        
        # Similarly:
        # M'[j, u] = M[j, u] + (a[j,j] - a[j,i])(b[u,u] - b[u,v])
        
        # Delta = M'[i, v] + M'[j, u] - M[i, u] - M[j, v]
        #       = M[i, v] + (a[i,i] - a[i,j])(b[v,v] - b[v,u]) + M[j, u] + (a[j,j] - a[j,i])(b[u,u] - b[u,v]) - M[i, u] - M[j, v]
        
        # Let's vectorize this.
        # p is current permutation.
        # u = p[i], v = p[j]
        
        # M_vals = M
        # b_uu = b[p[i], p[i]]
        # b_vv = b[p[j], p[j]]
        # b_uv = b[p[i], p[j]]
        # b_vu = b[p[j], p[i]]
        
        # Since b is symmetric, b_uv = b_vu.
        
        # delta_matrix[i, j] = M[i, p[j]] + M[j, p[i]] - M[i, p[i]] - M[j, p[j]] 
        #                      + (a[i,i] - a[i,j]) * (b[p[j], p[j]] - b[p[i], p[j]])
        #                      + (a[j,j] - a[j,i]) * (b[p[i], p[i]] - b[p[i], p[j]])
        
        # Note: diagonal elements i==j are not valid swaps.
        
        # Vectorized calculation:
        # p_idx = p
        # M_i_pj = M[:, p] # Shape (n, n), M[i, p[j]] is M_i_pj[i, j]
        # M_j_pi = M[:, p].T # Shape (n, n), M[j, p[i]] is M_j_pi[i, j]
        # M_i_pi = M[np.arange(n), p] # Shape (n,), M[i, p[i]]
        # M_j_pj = M[np.arange(n), p] # Same as above, M[j, p[j]]
        
        # Let's use broadcasting.
        # i_idx = np.arange(n)
        # M_ipj = M[:, p] # M[i, p[j]]
        # M_jpi = M[:, p].T # M[j, p[i]]
        # M_ii = M[i_idx, p] # M[i, p[i]]
        
        # b_pp = b[np.ix_(p, p)] # b[p[i], p[j]]
        # b_ii = b[p, p] # b[p[i], p[i]]
        
        # a_ii = a.diagonal()
        
        # Term 1: M_ipj + M_jpi - M_ii - M_ii (since M_ii[i] = M[i,p[i]] and M_ii[j] = M[j,p[j]], but we are summing over i,j pairs)
        # Actually, for pair (i,j), we subtract M[i,p[i]] and M[j,p[j]].
        # So base_delta = M[:, p] + M[:, p].T - M[i_idx, p][np.newaxis, :] - M[i_idx, p][:, np.newaxis]
        
        # Term 2: (a[i,i] - a[i,j]) * (b[p[j], p[j]] - b[p[i], p[j]])
        # a_ii_col = a_ii[:, np.newaxis]
        # a_ij = a
        # b_jj_row = b[p, p][np.newaxis, :] # b[p[j], p[j]]
        # b_ij = b[np.ix_(p, p)] # b[p[i], p[j]]
        # corr_2 = (a_ii_col - a) * (b_jj_row - b_ij)
        
        # Term 3: (a[j,j] - a[j,i]) * (b[p[i], p[i]] - b[p[i], p[j]])
        # a_jj_row = a_ii[np.newaxis, :]
        # a_ji = a.T
        # b_ii_col = b[p, p][:, np.newaxis]
        # b_ji = b[np.ix_(p, p)].T
        # corr_3 = (a_jj_row - a.T) * (b_ii_col - b_ji)
        
        # Total Delta = base_delta + corr_2 + corr_3
        
        # Mask out diagonal (i==j)
        # delta_matrix = base_delta + corr_2 + corr_3
        # np.fill_diagonal(delta_matrix, np.inf)
        
        # Find min
        # min_idx = np.unravel_index(np.argmin(delta_matrix), delta_matrix.shape)
        # i, j = min_idx
        
        # If delta < 0, apply swap
        # p[i], p[j] = p[j], p[i]
        # Update best if cost improves (we can just recompute cost or use delta)
        
        # Optimization: We can compute the full delta matrix in O(n^2) memory and time (vectorized).
        # n=150, so 22500 elements. Very fast.
        
        i_idx = np.arange(n)
        
        while time.time() < deadline:
            M = a @ b[:, p].T
            
            M_ipj = M[:, p]
            M_jpi = M[:, p].T
            M_ii = M[i_idx, p]
            
            b_pp = b[np.ix_(p, p)]
            b_ii = b[p, p]
            
            a_ii = a.diagonal()
            
            base_delta = M_ipj + M_jpi - M_ii[np.newaxis, :] - M_ii[:, np.newaxis]
            
            corr_2 = (a_ii[:, np.newaxis] - a) * (b_ii[np.newaxis, :] - b_pp)
            corr_3 = (a_ii[np.newaxis, :] - a.T) * (b_ii[:, np.newaxis] - b_pp.T)
            
            delta_matrix = base_delta + corr_2 + corr_3
            np.fill_diagonal(delta_matrix, np.inf)
            
            min_idx = np.unravel_index(np.argmin(delta_matrix), delta_matrix.shape)
            i, j = min_idx
            delta = delta_matrix[i, j]
            
            if delta >= 0:
                break # Local minimum found
            
            # Apply swap
            p[i], p[j] = p[j], p[i]
            
            # Update best
            # We can approximate new cost as best_cost + delta
            # But to be safe and account for any integer overflow or logic errors, 
            # we can periodically recompute. 
            # Given the exactness of the delta formula, best_cost + delta should be exact.
            best_cost += delta
            best_p = p.copy()
            
    return best_p.tolist()
