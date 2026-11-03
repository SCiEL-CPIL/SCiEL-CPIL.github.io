"""QAP solver for tho150 using vectorized local search with efficient swap evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    """Solve QAP: assign facilities to locations to minimize sum a[i,j]*b[p[i],p[j]].
    
    Args:
        a: flow matrix (n x n) int64
        b: distance matrix (n x n) int64
    Returns:
        p: permutation list where facility i goes to location p[i]
    """
    n = len(a)
    deadline = time.time() + 112  # Safety margin before 150s kill

    # Convert to float for faster arithmetic
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    # Precompute row and column sums of a
    a_row_sums = a_f.sum(axis=1)
    a_col_sums = a_f.sum(axis=0)
    # Combined coefficient: c[k] = sum_j a[j,k] + sum_j a[k,j]
    c = a_col_sums + a_row_sums

    def compute_cost(p):
        """Compute exact cost of permutation p using integer arithmetic."""
        M = b[np.ix_(p, p)]
        return int(np.sum(a * M))

    def compute_cost_float(p):
        """Compute cost using float (faster, for delta tracking)."""
        M = b_f[np.ix_(p, p)]
        return float(np.sum(a_f * M))

    def get_all_swap_deltas(p):
        """Compute cost deltas for ALL swaps (i,j) with i<j.
        
        Returns a 2D array D where D[i,j] = cost change when swapping p[i] and p[j] (for i<j).
        Uses the identity: cost = sum_k c[k]*b[p[k], p_i] ... 
        Actually: cost = sum_{i,j} a[i,j]*b[p[i],p[j]]
        = sum_k [sum_j a[k,j]*b[p[k],p[j]] + sum_j a[j,k]*b[p[j],p[k]]]
        = sum_k c[k]*b[p[k], p_k] + sum_{k != j} (a[k,j]+a[j,k])*b[p[k],p[j]]
        
        Simpler: define for each position k, its "contribution vector":
        The total cost = sum_k sum_j a[k,j] * b[p[k], p[j]]
        
        When we swap positions i and j (with values pi=p[i], pj=p[j]):
        Only terms involving p[i] or p[j] change.
        Change = sum_k a[k,i]*(b[p[k],pj]-b[p[k],pi]) + a[i,k]*(b[pj,p[k]]-b[pi,p[k]])
               + sum_k a[k,j]*(b[p[k],pi]-b[p[k],pj]) + a[j,k]*(b[pi,p[k]]-b[pj,p[k]])
               - a[i,i]*(b[pi,pi]-b[pj,pj]) - a[j,j]*(b[pj,pj]-b[pi,pi])  # diagonal fix
               - a[i,j]*(b[pi,pj]-b[pj,pi]) - a[j,i]*(b[pj,pi]-b[pi,pj])  # cross fix
        
        Since b is symmetric: b[pi,pj]=b[pj,pi], so cross terms:
        a[i,j]*(b[pi,pj]-b[pj,pi]) = 0, and a[j,i]*(b[pj,pi]-b[pi,pj]) = 0.
        Diagonal: the terms a[i,i]*b[pi,pi] and a[j,j]*b[pj,pj] were counted in the sums above
        (when k=i in first sum, k=j in third sum). Let me be careful.
        
        Let me use a cleaner formulation.
        Total cost C = sum_{u,v} a[u,v] * b[p[u], p[v]]
        
        Swap positions i and j. Let pi = p[i], pj = p[j].
        After swap: p'[i]=pj, p'[j]=pi, p'[k]=p[k] for k!=i,j.
        
        C' - C = sum_{u,v} a[u,v] * (b[p'[u],p'[v]] - b[p[u],p[v]])
        
        Only terms where u or v is i or j change.
        = sum_v a[i,v]*(b[pj, p'[v]] - b[pi, p[v]]) 
          + sum_u a[u,i]*(b[p'[u], pj] - b[p[u], pi])
          + sum_v a[j,v]*(b[pi, p'[v]] - b[pj, p[v]])
          + sum_u a[u,j]*(b[p'[u], pi] - b[p[u], pj])
        
        For v not in {i,j}: p'[v]=p[v], so:
        First sum (v not i,j): a[i,v]*(b[pj,p[v]] - b[pi,p[v]]) = (b[pj,p[v]]-b[pi,p[v]])*a[i,v]
        Second sum (u not i,j): a[u,i]*(b[p[u],pj] - b[p[u],pi])
        Third sum (v not i,j): a[j,v]*(b[pi,p[v]] - b[pj,p[v]])
        Fourth sum (u not i,j): a[u,j]*(b[p[u],pi] - b[p[u],pj])
        
        For v=i in first sum: a[i,i]*(b[pj,pj] - b[pi,pi])
        For v=j in first sum: a[i,j]*(b[pj,pi] - b[pi,pj]) = 0 (symmetric b)
        For u=i in second sum: a[i,i]*(b[pj,pj] - b[pi,pi])
        For u=j in second sum: a[j,i]*(b[pi,pj] - b[pj,pi]) = 0
        For v=i in third sum: a[j,i]*(b[pi,pj] - b[pj,pi]) = 0
        For v=j in third sum: a[j,j]*(b[pi,pi] - b[pj,pj])
        For u=i in fourth sum: a[i,j]*(b[pj,pi] - b[pi,pj]) = 0
        For u=j in fourth sum: a[j,j]*(b[pi,pi] - b[pj,pj])
        
        So total:
        C'-C = sum_{k not i,j} [ a[i,k]*(b[pj,p[k]]-b[pi,p[k]]) + a[k,i]*(b[p[k],pj]-b[p[k],pi])
                                 + a[j,k]*(b[pi,p[k]]-b[pj,p[k]]) + a[k,j]*(b[p[k],pi]-b[p[k],pj]) ]
              + 2*a[i,i]*(b[pj,pj]-b[pi,pi]) + 2*a[j,j]*(b[pi,pi]-b[pj,pj])
        
        = sum_{k not i,j} [ (a[i,k]-a[j,k])*(b[pj,p[k]]-b[pi,p[k]]) 
                           + (a[k,i]-a[k,j])*(b[p[k],pj]-b[p[k],pi]) ]
              + 2*(a[i,i]-a[j,j])*(b[pj,pj]-b[pi,pi])
        
        This is still O(n) per pair. For all pairs, O(n^3). Too slow for n=150 in tight loop.
        
        Let me use a different approach: precompute for each position i, the vector 
        r_i = a[i,:] (row) and c_i = a[:,i] (col).
        
        For a fixed i, the delta for swapping i with j is:
        delta(i,j) = (r_i - r_j) @ (b_f[pj, p] - b_f[pi, p]) 
                   + (c_i - c_j) @ (b_f[p, pj] - b_f[p, pi])
                   + 2*(a[i,i]-a[j,j])*(b_f[pj,pj]-b_f[pi,pi])
                   - (r_i - r_j)[i] * (b_f[pj,p[i]]-b_f[pi,p[i]])  # remove k=i term
                   - (c_i - c_j)[i] * (b_f[p[i],pj]-b_f[p[i],pi])  # remove k=i term
                   - (r_i - r_j)[j] * (b_f[pj,p[j]]-b_f[pi,p[j]])  # remove k=j term
                   - (c_i - c_j)[j] * (b_f[p[j],pj]-b_f[p[j],pi])  # remove k=j term
        
        Hmm, this is getting complex. Let me just do the O(n) per i, O(n^2) total per scan.
        """
        # For each i, compute deltas for all j > i
        # delta(i,j) = sum_{k != i,j} [(a[i,k]-a[j,k])*(b[pj,pk]-b[pi,pk]) + (a[k,i]-a[k,j])*(b[pk,pj]-b[pk,pi])]
        #            + 2*(a[i,i]-a[j,j])*(b[pj,pj]-b[pi,pi])
        #
        # Let M = b_f[np.ix_(p,p)] so M[j,k] = b[p[j],p[k]]
        M = b_f[np.ix_(p, p)]
        # M[j,:] = b_f[p[j], p]
        # M[:,j] = b_f[p, p[j]]
        
        deltas = np.zeros((n, n), dtype=np.float64)
        
        for i in range(n):
            if time.time() > deadline:
                break
            pi = p[i]
            row_i = a_f[i, :]
            col_i = a_f[:, i]
            b_pi_p = M[i, :]  # b[pi, p[k]] for all k
            b_p_pi = M[:, i]  # b[p[k], pi] for all k
            
            for j in range(i+1, n):
                pj = p[j]
                row_j = a_f[j, :]
                col_j = a_f[:, j]
                
                # Sum over k != i,j
                # (row_i[k]-row_j[k])*(M[j,k]-M[i,k]) + (col_i[k]-col_j[k])*(M[k,j]-M[k,i])
                diff_row = row_i - row_j
                diff_col = col_i - col_j
                diff_M_row = M[j, :] - M[i, :]  # b[pj,p[k]] - b[pi,p[k]]
                diff_M_col = M[:, j] - M[:, i]  # b[p[k],pj] - b[p[k],pi]
                
                # Full sum includes k=i and k=j, need to exclude
                total = np.dot(diff_row, diff_M_row) + np.dot(diff_col, diff_M_col)
                
                # Remove k=i term:
                # diff_row[i]*diff_M_row[i] + diff_col[i]*diff_M_col[i]
                # diff_M_row[i] = M[j,i]-M[i,i] = b[pj,pi]-b[pi,pi]
                # diff_M_col[i] = M[i,j]-M[i,i] = b[pi,pj]-b[pi,pi] = b[pj,pi]-b[pi,pi] (symmetric)
                # So both are b[pj,pi]-b[pi,pi]
                ki_term = diff_row[i]*(b_f[pj,pi]-b_f[pi,pi]) + diff_col[i]*(b_f[pj,pi]-b_f[pi,pi])
                
                # Remove k=j term:
                # diff_row[j]*diff_M_row[j] + diff_col[j]*diff_M_col[j]
                # diff_M_row[j] = M[j,j]-M[i,j] = b[pj,pj]-b[pi,pj]
                # diff_M_col[j] = M[j,j]-M[i,j] = b[pj,pj]-b[pi,pj] (symmetric)
                kj_term = diff_row[j]*(b_f[pj,pj]-b_f[pi,pj]) + diff_col[j]*(b_f[pj,pj]-b_f[pi,pj])
                
                total -= ki_term + kj_term
                
                # Diagonal correction
                diag = 2.0*(a_f[i,i]-a_f[j,j])*(b_f[pj,pj]-b_f[pi,pi])
                
                deltas[i, j] = total + diag
        
        return deltas

    def local_search(p, deadline):
        """Steepest descent local search. Returns improved permutation and cost."""
        cost = compute_cost_float(p)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            
            D = get_all_swap_deltas(p)
            if time.time() > deadline:
                break
            
            # Find best swap
            # Only consider upper triangle
            D_lower = np.triu(D, k=1)
            best_val = np.min(D_lower)
            if best_val < 0:
                # Find the position
                mask = (D_lower == best_val)
                rows, cols = np.where(mask)
                if len(rows) > 0:
                    best_i = rows[0]
                    best_j = cols[0]
                    best_delta = best_val
            
            if best_i >= 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                cost += best_delta
                improved = True
        
        return p, cost

    def spectral_initialization(a_mat, b_mat, rng):
        """Spectral method for QAP initialization."""
        # Eigendecomposition approach
        # For QAP min sum a[i,j]*b[p[i],p[j]], a common spectral method:
        # Compute eigenvectors of a and b, match them.
        try:
            # Use the fact that for symmetric matrices, 
            # the optimal assignment aligns eigenvectors
            # We use the "rank" of eigenvectors
            w_a, v_a = np.linalg.eigh(a_mat)
            w_b, v_b = np.linalg.eigh(b_mat)
            
            # Sort by eigenvalue magnitude (descending)
            idx_a = np.argsort(-np.abs(w_a))
            idx_b = np.argsort(-np.abs(w_b))
            
            # Match: facility i (eigenvector idx_a[i]) to location j (eigenvector idx_b[i])
            # Actually, we want to match the principal components
            # Let's use the sign-matching approach
            p = np.zeros(n, dtype=np.int64)
            for i in range(n):
                p[i] = idx_b[i]
            
            # This might not be great. Let's use a simpler rank-based approach
            # combined with spectral
            a_row_sums = a_mat.sum(axis=1)
            b_row_sums = b_mat.sum(axis=1)
            
            # Rank by row sum
            fac_rank = np.argsort(-a_row_sums)
            loc_rank = np.argsort(b_row_sums)
            
            p2 = np.empty(n, dtype=np.int64)
            p2[fac_rank] = loc_rank
            
            # Average the two
            # Just use rank-based for reliability
            return p2
        except:
            # Fallback
            a_row_sums = a_mat.sum(axis=1)
            b_row_sums = b_mat.sum(axis=1)
            fac_rank = np.argsort(-a_row_sums)
            loc_rank = np.argsort(b_row_sums)
            p = np.empty(n, dtype=np.int64)
            p[fac_rank] = loc_rank
            return p

    def make_initial_solution(method, rng):
        """Create an initial permutation using different strategies."""
        a_row_sums = a_f.sum(axis=1)
        b_row_sums = b_f.sum(axis=1)
        
        if method == 0:
            # Rank matching: heavy flow -> low distance sum location
            heavy = np.argsort(-a_row_sums, kind='stable')
            central = np.argsort(b_row_sums, kind='stable')
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
        elif method == 1:
            # Reverse rank
            heavy = np.argsort(a_row_sums, kind='stable')
            central = np.argsort(-b_row_sums, kind='stable')
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
        elif method == 2:
            # Random
            p = rng.permutation(n).astype(np.int64)
        elif method == 3:
            # Sort by a_row_sums / b_row_sums ratio
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(b_row_sums > 0, a_row_sums / np.maximum(b_row_sums, 1), 0)
            fac_order = np.argsort(-ratio, kind='stable')
            loc_order = np.argsort(b_row_sums, kind='stable')
            p = np.empty(n, dtype=np.int64)
            p[fac_order] = loc_order
        elif method == 4:
            # Spectral
            p = spectral_initialization(a_f, b_f, rng)
        else:
            p = rng.permutation(n).astype(np.int64)
        return p

    # Multiple initializations with ILS
    best_p = None
    best_cost = np.inf
    
    rng = np.random.default_rng(42)
    
    # Try several different initial solutions
    n_init = 10
    for init_idx in range(n_init):
        if time.time() > deadline:
            break
        
        method = init_idx % 5
        p = make_initial_solution(method, rng)
        
        # Local search
        p, cost = local_search(p, deadline)
        
        cost_int = compute_cost(p)
        if cost_int < best_cost:
            best_cost = cost_int
            best_p = p.copy()
        
        # ILS: perturb and search again if time permits
        if time.time() < deadline - 40:
            for ils_iter in range(8):
                if time.time() > deadline:
                    break
                # Perturb: random swaps
                n_swaps = rng.integers(3, 15)
                p_perturbed = p.copy()
                for _ in range(n_swaps):
                    i, j = rng.integers(0, n, size=2)
                    if i != j:
                        p_perturbed[i], p_perturbed[j] = p_perturbed[j], p_perturbed[i]
                
                p_perturbed, cost_p = local_search(p_perturbed, deadline)
                
                cost_p_int = compute_cost(p_perturbed)
                if cost_p_int < best_cost:
                    best_cost = cost_p_int
                    best_p = p_perturbed.copy()
                    p = p_perturbed.copy()
                    cost = cost_p
                elif cost_p < cost:
                    p = p_perturbed.copy()
                    cost = cost_p
    
    # Final high-precision polish: do a few more local search passes on the best solution
    if best_p is not None and time.time() < deadline - 5:
        p_polish = best_p.copy()
        p_polish, cost_polish = local_search(p_polish, deadline)
        cost_polish_int = compute_cost(p_polish)
        if cost_polish_int < best_cost:
            best_cost = cost_polish_int
            best_p = p_polish.copy()
    
    # Final verification
    if best_p is None:
        best_p = make_initial_solution(0, rng)
    
    final_cost = compute_cost(best_p)
    if final_cost < best_cost:
        best_cost = final_cost
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
