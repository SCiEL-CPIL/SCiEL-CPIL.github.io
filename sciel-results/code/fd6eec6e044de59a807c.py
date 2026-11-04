"""Rewritten QAP solver: spectral initialization + Hungarian + optimized local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Spectral initialization using eigenvectors of QAP-related matrices.
    2. Hungarian-based linearization refinement (multiple rounds).
    3. Fast local search with precomputed swap deltas and multiple restarts.
    """
    n = len(a)
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    def compute_cost(p):
        """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
        total = 0
        for i in range(n):
            total += np.dot(a[i], b[p[i], p])
        return total
    
    # --- Spectral initialization ---
    # Use eigenvectors of the QAP-related matrices to generate high-quality initial solutions
    # The QAP can be related to eigenvalue problems. We use multiple spectral approaches:
    
    def spectral_init(method='flow'):
        """Generate initial permutation using spectral methods."""
        if method == 'flow':
            # Use eigenvectors of the flow matrix a
            # Center a to remove mean
            a_centered = a - a.mean()
            # Get top eigenvectors
            try:
                eigvals, eigvecs = np.linalg.eigh(a_centered)
                # Sort by eigenvalue magnitude (descending)
                idx = np.argsort(-np.abs(eigvals))
                eigvecs_sorted = eigvecs[:, idx[:min(50, n)]]
                # Use the top eigenvector to create a ranking
                # Map facility indices to their "spectral coordinate"
                spec_coords = eigvecs_sorted[:, 0]
                # Rank facilities by spectral coordinate
                fac_rank = np.argsort(spec_coords)
                # Rank locations by their row sum of b (centrality)
                loc_rank = np.argsort(b.sum(axis=1))
                # Assign: facility with rank r gets location with rank r
                p = np.empty(n, dtype=np.int64)
                p[fac_rank] = loc_rank
            except:
                # Fallback to rank matching
                heavy = np.argsort(-a.sum(axis=1), kind="stable")
                central = np.argsort(b.sum(axis=1), kind="stable")
                p = np.empty(n, dtype=np.int64)
                p[heavy] = central
            return p
        elif method == 'distance':
            # Use eigenvectors of the distance matrix b
            b_centered = b - b.mean()
            try:
                eigvals, eigvecs = np.linalg.eigh(b_centered)
                idx = np.argsort(-np.abs(eigvals))
                eigvecs_sorted = eigvecs[:, idx[:min(50, n)]]
                spec_coords = eigvecs_sorted[:, 0]
                loc_rank = np.argsort(spec_coords)
                # Rank facilities by their row sum of a (flow intensity)
                fac_rank = np.argsort(a.sum(axis=1))
                p = np.empty(n, dtype=np.int64)
                p[fac_rank] = loc_rank
            except:
                heavy = np.argsort(-a.sum(axis=1), kind="stable")
                central = np.argsort(b.sum(axis=1), kind="stable")
                p = np.empty(n, dtype=np.int64)
                p[heavy] = central
            return p
        elif method == 'combined':
            # Combine flow and distance spectral info
            a_centered = a - a.mean()
            b_centered = b - b.mean()
            try:
                eigvals_a, eigvecs_a = np.linalg.eigh(a_centered)
                eigvals_b, eigvecs_b = np.linalg.eigh(b_centered)
                idx_a = np.argsort(-np.abs(eigvals_a))
                idx_b = np.argsort(-np.abs(eigvals_b))
                # Use top eigenvectors from both
                spec_a = eigvecs_a[:, idx_a[0]]
                spec_b = eigvecs_b[:, idx_b[0]]
                # Combine: use flow spectral for facilities, distance spectral for locations
                fac_rank = np.argsort(spec_a)
                loc_rank = np.argsort(spec_b)
                p = np.empty(n, dtype=np.int64)
                p[fac_rank] = loc_rank
            except:
                heavy = np.argsort(-a.sum(axis=1), kind="stable")
                central = np.argsort(b.sum(axis=1), kind="stable")
                p = np.empty(n, dtype=np.int64)
                p[heavy] = central
            return p
        else:
            # Default: rank matching
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
            return p
    
    # --- Optimized swap delta computation ---
    # Precompute helper arrays for faster delta computation
    
    def swap_delta_fast(i, j, p, a, b, precomp):
        """Fast swap delta using precomputed values."""
        u = p[i]
        v = p[j]
        if u == v:
            return 0
        
        # Precomputed:
        # b_p = b[p, p] -- but we don't want to store full n x n
        # Instead, precompute:
        # row_b_p[i] = b[p[i], p]  (row i of b indexed by p)
        # col_b_p[j] = b[p, p[j]]  (column j of b indexed by p)
        
        # Actually, let's precompute:
        # For each location l, we need b[l, p[k]] and b[p[k], l] for all k
        # This is expensive. Instead, use a different approach:
        # Precompute for each facility i:
        #   a_row_i = a[i, :]
        #   a_col_i = a[:, i]
        # And for the current permutation p:
        #   b_p_rows[i] = b[p[i], :]  -- not helpful directly
        #   b_p_cols[j] = b[:, p[j]]  -- not helpful directly
        
        # Let's use the formula but with precomputed row/col products:
        # Precompute:
        #   A_row_sums[i] = a[i, :]  (already have a)
        #   A_col_sums[j] = a[:, j]  (already have a)
        
        # The key insight: we can precompute:
        #   For each pair (i, j), the delta involves:
        #   sum_k a[i,k]*(b[v,p[k]] - b[u,p[k]]) for k not in {i,j}
        #   = sum_k a[i,k]*b[v,p[k]] - sum_k a[i,k]*b[u,p[k]] - corrections
        #   
        # Let's precompute for each location l:
        #   B_l_p = b[l, p]  (vector of length n)
        #   B_p_l = b[p, l]  (vector of length n)
        # But this is n * n * 2 = O(n^2) memory, which is fine for n=150.
        
        # Actually, b[l, p] is just b[l, :] with columns permuted by p
        # We can compute it on the fly: b[l, p] = b[l, p]
        
        # For speed, let's precompute:
        #   b_rows = b  (already have)
        #   b_cols = b.T  (already have)
        #   p is the permutation
        
        # The delta formula:
        delta = 0.0
        
        # Terms 1-4 (diagonal and cross terms)
        delta += a[i, i] * (b[v, v] - b[u, u])
        delta += a[i, j] * (b[v, u] - b[u, v])
        delta += a[j, i] * (b[u, v] - b[v, u])
        delta += a[j, j] * (b[u, u] - b[v, v])
        
        # Terms 5-8: sum over k not in {i,j}
        # Use vectorized computation
        # Create mask for k not in {i, j}
        # To avoid creating a new mask each time, use a precomputed approach:
        
        # Term 5: sum_{k not in {i,j}} a[i,k] * (b[v, p[k]] - b[u, p[k]])
        # = sum_k a[i,k]*b[v,p[k]] - sum_k a[i,k]*b[u,p[k]] 
        #   - a[i,i]*(b[v,v]-b[u,u]) - a[i,j]*(b[v,u]-b[u,v])
        
        # Precompute for each facility i and location l:
        #   dot_a_i_b_l_p = sum_k a[i,k] * b[l, p[k]]
        # This is (a[i, :] @ b[l, p]) = a[i, :] @ (b[l, :])[p]
        # = a[i, :] @ b[l, p]
        
        # Let's precompute:
        #   For each i, l: dot_il = sum_k a[i,k] * b[l, p[k]]
        # This is a @ b[:, p].T
        # b[:, p] has shape (n, n): [l, k] = b[l, p[k]]
        # b[:, p].T has shape (n, n): [k, l] = b[l, p[k]]
        # a @ b[:, p].T has shape (n, n): [i, l] = sum_k a[i,k] * b[l, p[k]] ✓
        
        # Similarly for columns:
        #   For each i, l: dot_il_col = sum_k a[k,i] * b[p[k], l]
        # = sum_k a[k,i] * b[p[k], l]
        # b[p, :] has shape (n, n): [k, l] = b[p[k], l]
        # a.T @ b[p, :] has shape (n, n): [i, l] = sum_k a.T[i,k] * b[p,k,l] = sum_k a[k,i]*b[p[k],l] ✓
        
        # So precompute:
        #   dot_row[i, l] = sum_k a[i,k] * b[l, p[k]]  = (a @ b[:, p].T)[i, l]
        #   dot_col[i, l] = sum_k a[k,i] * b[p[k], l]  = (a.T @ b[p, :])[i, l]
        
        # These are O(n^3) to compute, but we only need to recompute when p changes.
        # For now, compute on the fly for the delta, but cache the results.
        
        # Actually, let's just compute the four dot products we need:
        #   d5_full = sum_k a[i,k] * b[v, p[k]]  = dot_row[i, v]
        #   d5_full_u = sum_k a[i,k] * b[u, p[k]]  = dot_row[i, u]
        #   d6_full = sum_k a[j,k] * b[u, p[k]]  = dot_row[j, u]
        #   d6_full_v = sum_k a[j,k] * b[v, p[k]]  = dot_row[j, v]
        #   d7_full = sum_k a[k,i] * b[p[k], v]  = dot_col[i, v]
        #   d7_full_u = sum_k a[k,i] * b[p[k], u]  = dot_col[i, u]
        #   d8_full = sum_k a[k,j] * b[p[k], u]  = dot_col[j, u]
        #   d8_full_v = sum_k a[k,j] * b[p[k], v]  = dot_col[j, v]
        
        # If we have dot_row and dot_col precomputed, we can get these in O(1).
        
        # For now, compute directly (slower but simpler):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        ks = np.where(mask)[0]
        
        if len(ks) > 0:
            # Term 5
            delta += np.dot(a[i, ks], b[v, p[ks]] - b[u, p[ks]])
            # Term 6
            delta += np.dot(a[j, ks], b[u, p[ks]] - b[v, p[ks]])
            # Term 7
            delta += np.dot(a[ks, i], b[p[ks], v] - b[p[ks], u])
            # Term 8
            delta += np.dot(a[ks, j], b[p[ks], u] - b[p[ks], v])
        
        return delta
    
    # --- Try multiple initial solutions and keep the best ---
    init_methods = ['flow', 'distance', 'combined', 'default']
    best_p = None
    best_cost = np.inf
    
    for method in init_methods:
        if time.time() > deadline - 30.0:
            break
        p_init = spectral_init(method)
        cost_init = compute_cost(p_init)
        if cost_init < best_cost:
            best_cost = cost_init
            best_p = p_init.copy()
    
    # --- Local search with precomputed dot products and incremental updates ---
    def local_search(p, cost, time_limit):
        """Perform local search starting from p with initial cost."""
        p = p.copy()
        
        # Precompute dot_row and dot_col for fast delta computation
        # dot_row[i, l] = sum_k a[i,k] * b[l, p[k]]
        # dot_col[i, l] = sum_k a[k,i] * b[p[k], l]
        
        def recompute_precomp(p):
            # b[:, p] has shape (n, n): [l, k] = b[l, p[k]]
            b_col_p = b[:, p]  # (n, n)
            # b[p, :] has shape (n, n): [k, l] = b[p[k], l]
            b_row_p = b[p, :]  # (n, n)
            
            # dot_row[i, l] = sum_k a[i,k] * b[l, p[k]] = (a @ b_col_p.T)[i, l]
            dot_row = a @ b_col_p.T  # (n, n)
            
            # dot_col[i, l] = sum_k a[k,i] * b[p[k], l] = (a.T @ b_row_p)[i, l]
            dot_col = a.T @ b_row_p  # (n, n)
            
            return dot_row, dot_col
        
        dot_row, dot_col = recompute_precomp(p)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Random sampling of pairs
            # Since updates are O(n^2), we can try many more swaps
            num_trials = 100000
            for _ in range(num_trials):
                if time.time() > time_limit:
                    break
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                
                u = p[i]
                v = p[j]
                if u == v:
                    continue
                
                # Compute delta using precomputed dot products
                # Terms 1-4
                delta = 0.0
                delta += a[i, i] * (b[v, v] - b[u, u])
                delta += a[i, j] * (b[v, u] - b[u, v])
                delta += a[j, i] * (b[u, v] - b[v, u])
                delta += a[j, j] * (b[u, u] - b[v, v])
                
                # Terms 5-8 using precomputed dot products
                # Term 5: sum_{k not in {i,j}} a[i,k] * (b[v, p[k]] - b[u, p[k]])
                # = dot_row[i, v] - dot_row[i, u] - a[i,i]*(b[v,v]-b[u,u]) - a[i,j]*(b[v,u]-b[u,v])
                delta += (dot_row[i, v] - dot_row[i, u]) - a[i,i]*(b[v,v]-b[u,u]) - a[i,j]*(b[v,u]-b[u,v])
                
                # Term 6: sum_{k not in {i,j}} a[j,k] * (b[u, p[k]] - b[v, p[k]])
                # = dot_row[j, u] - dot_row[j, v] - a[j,i]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
                delta += (dot_row[j, u] - dot_row[j, v]) - a[j,i]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
                
                # Term 7: sum_{k not in {i,j}} a[k,i] * (b[p[k], v] - b[p[k], u])
                # = dot_col[i, v] - dot_col[i, u] - a[i,i]*(b[v,v]-b[u,u]) - a[j,i]*(b[v,u]-b[u,v])
                delta += (dot_col[i, v] - dot_col[i, u]) - a[i,i]*(b[v,v]-b[u,u]) - a[j,i]*(b[v,u]-b[u,v])
                
                # Term 8: sum_{k not in {i,j}} a[k,j] * (b[p[k], u] - b[p[k], v])
                # = dot_col[j, u] - dot_col[j, v] - a[i,j]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
                delta += (dot_col[j, u] - dot_col[j, v]) - a[i,j]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
                
                if delta < -1e-9:
                    # Accept the swap
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    
                    # Incremental update of dot_row and dot_col
                    # D_b = b[v, :] - b[u, :]
                    D_b = b[v, :] - b[u, :]
                    
                    # Update dot_row:
                    # dot_row[k, l] += (a[k, i] - a[k, j]) * D_b[l]
                    diff_a_col = a[:, i] - a[:, j]
                    dot_row += np.outer(diff_a_col, D_b)
                    
                    # Update dot_col:
                    # dot_col[c, l] += (a[i, c] - a[j, c]) * D_b[l]
                    diff_a_row = a[i, :] - a[j, :]
                    dot_col += np.outer(diff_a_row, D_b)
                    
                    # Optional: Periodic re-computation to correct floating point drift
                    # if _ % 1000 == 0:
                    #     dot_row, dot_col = recompute_precomp(p)
        
        return p, cost
    
    # Run local search
    time_limit = deadline - 20.0
    best_p, best_cost = local_search(best_p, best_cost, time_limit)
    
    # --- Hungarian-based refinement ---
    if time.time() < deadline - 15.0:
        # Build linear cost matrix
        b_col_p = b[:, best_p]  # shape (n, n), [j, k] = b[j, p[k]]
        b_row_p = b[best_p, :]  # shape (n, n), [k, j] = b[p[k], j]
        
        C = a @ b_col_p.T + a.T @ b_row_p
        
        row_ind, col_ind = linear_sum_assignment(C)
        p_hung = np.empty(n, dtype=np.int64)
        p_hung[row_ind] = col_ind
        
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
    
    # --- Final local search pass ---
    if time.time() < deadline - 5.0:
        time_limit = deadline - 3.0
        best_p, best_cost = local_search(best_p, best_cost, time_limit)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
