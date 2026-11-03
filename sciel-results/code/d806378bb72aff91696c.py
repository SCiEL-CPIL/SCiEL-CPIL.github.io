"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Ensure int64 for exact arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    
    # Check symmetry
    is_symmetric_a = np.array_equal(a, a.T)
    is_symmetric_b = np.array_equal(b, b.T)
    use_fast_path = is_symmetric_a and is_symmetric_b
    
    def calc_cost(p):
        # sum_{i,j} a[i,j] * b[p[i], p[j]]
        b_perm = b[:, p][:, p]
        return int(np.sum(a * b_perm))
    
    def init_rank_matching():
        """Gravity heuristic: match heavy facilities to central locations."""
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        
        heavy_order = np.argsort(-a_row_sums, kind="stable")
        central_order = np.argsort(b_row_sums, kind="stable")
        
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order
        return p
    
    def vec_swap_delta(p, B_p):
        """
        Compute delta for all swaps (i,j) where i < j.
        Returns array of deltas, shape (n, n) with upper triangle valid.
        """
        if use_fast_path:
            # For symmetric matrices:
            # delta(i,j) = 2 * sum_{k != i,j} (a[i,k] - a[j,k]) * (b[p[j], p[k]] - b[p[i], p[k]])
            # Vectorized: for each i, compute contribution to all j
            # Let u = p[i], v = p[j]
            # diff_a[i,j] = a[i,:] - a[j,:]  (n x n)
            # diff_b[i,j] = b[v,:] - b[u,:]  where u=p[i], v=p[j]
            
            # Precompute: for each location x, B_p[x] = b[x, p[0..n-1]]
            # We want for each pair (i,j): dot(a[i]-a[j], b[p[j],:]-b[p[i],:])
            # = dot(a[i], b[p[j],:]) - dot(a[i], b[p[i],:]) - dot(a[j], b[p[j],:]) + dot(a[j], b[p[i],:])
            
            # Let A_dot_Bx[i, x] = dot(a[i], b[x, p[:]]) = (a @ b[:, p])[i, x]
            # Actually: dot(a[i], b[x, p[:]]) = sum_k a[i,k] * b[x, p[k]]
            # This is (a @ b.T)[i, x] where we index b by p... 
            # Let's define M[i, x] = sum_k a[i,k] * b[x, p[k]] = (a @ b[:, p].T)[i, x]
            # Wait: b[:, p] has shape (n, n) where b[:, p][x, k] = b[x, p[k]]
            # So M[i, x] = sum_k a[i,k] * b[x, p[k]] = (a @ b[:, p].T)[i, x]
            # Hmm, a is (n,n), b[:, p] is (n,n)
            # (a @ b[:, p].T) [i, x] = sum_k a[i,k] * b[:, p].T[k, x] = sum_k a[i,k] * b[:, p][x, k] = sum_k a[i,k] * b[x, p[k]]
            # Yes! So M = a @ b[:, p].T, shape (n, n)
            
            M = a @ B_p.T  # M[i, x] = sum_k a[i,k] * b[x, p[k]]
            
            # For swap (i,j) with u=p[i], v=p[j]:
            # dot(a[i]-a[j], b[v,:]-b[u,:]) = M[i,v] - M[i,u] - M[j,v] + M[j,u]
            # But this includes k=i and k=j terms which need correction.
            # Actually for the full sum including all k:
            # sum_k (a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]])
            # = sum_k a[i,k]*b[v,p[k]] - sum_k a[i,k]*b[u,p[k]] - sum_k a[j,k]*b[v,p[k]] + sum_k a[j,k]*b[u,p[k]]
            # = M[i,v] - M[i,u] - M[j,v] + M[j,u]
            
            # This is the full dot product including k=i and k=j.
            # For symmetric case, delta = 2 * (full_dot - term_i - term_j)
            # where term_i and term_j correct for the k=i and k=j terms.
            
            # Let's compute the full matrix of raw dots:
            # raw[i,j] = M[i, p[j]] - M[i, p[i]] - M[j, p[j]] + M[j, p[i]]
            Mp = M[:, p]  # Mp[i, j] = M[i, p[j]] = sum_k a[i,k]*b[p[j],p[k]]
            
            # raw[i,j] = Mp[i,j] - Mp[i,i] - Mp[j,j] + Mp[j,i]
            raw = Mp - Mp[np.arange(n), np.arange(n)][:, np.newaxis] - Mp[np.arange(n), np.arange(n)][np.newaxis, :] + Mp.T
            
            # Now compute corrections for k=i and k=j
            # term_i for pair (i,j): (a[i,i] - a[j,i]) * (b[v, u] - b[u, u]) where u=p[i], v=p[j]
            # = (a[i,i] - a[j,i]) * (b[p[j], p[i]] - b[p[i], p[i]])
            # term_j for pair (i,j): (a[i,j] - a[j,j]) * (b[v, v] - b[u, v])
            # = (a[i,j] - a[j,j]) * (b[p[j], p[j]] - b[p[i], p[j]])
            
            # Build correction matrices
            # term_i[i,j] = (a[i,i] - a[j,i]) * (b[p[j], p[i]] - b[p[i], p[i]])
            # Let diag_a = a.diagonal()  # diag_a[i] = a[i,i]
            # bpp = b[p[:,None], p[None,:]]  # bpp[i,j] = b[p[i], p[j]]
            
            diag_a = np.diag(a)
            bpp = b[np.ix_(p, p)]  # bpp[i,j] = b[p[i], p[j]]
            
            # term_i[i,j] = (diag_a[i] - a[j,i]) * (bpp[j,i] - bpp[i,i])
            # term_j[i,j] = (a[i,j] - diag_a[j]) * (bpp[j,j] - bpp[i,j])
            
            term_i = (diag_a[:, np.newaxis] - a.T) * (bpp.T - np.diag(bpp)[:, np.newaxis])
            term_j = (a - diag_a[np.newaxis, :]) * (np.diag(bpp)[np.newaxis, :] - bpp)
            
            # full_dot excluding k=i,j:
            # For pair (i,j), the terms k=i and k=j in the sum are:
            # k=i: (a[i,i]-a[j,i])*(b[v,p[i]]-b[u,p[i]]) = (a[i,i]-a[j,i])*(b[v,u]-b[u,u]) = term_i[i,j]
            # k=j: (a[i,j]-a[j,j])*(b[v,p[j]]-b[u,p[j]]) = (a[i,j]-a[j,j])*(b[v,v]-b[u,v]) = term_j[i,j]
            
            # So corrected = raw - term_i - term_j
            corrected = raw - term_i - term_j
            
            # delta = 2 * corrected
            deltas = 2 * corrected
            
            # Set diagonal and lower triangle to inf (we only consider i < j)
            deltas = np.triu(deltas, k=1)
            # Replace 0s on diagonal with inf
            np.fill_diagonal(deltas, np.inf)
            
            return deltas
        else:
            # General (non-symmetric) case - more complex
            # delta = sum_{k!=i,j}(a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]]) 
            #       + sum_{k!=i,j}(a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
            #       + direct terms
            # This is harder to vectorize efficiently. Fall back to pairwise.
            deltas = np.full((n, n), np.inf, dtype=np.int64)
            for i in range(n):
                u = p[i]
                for j in range(i + 1, n):
                    v = p[j]
                    diff_a_row = a[i] - a[j]
                    diff_b_row = B_p[v] - B_p[u]
                    sum1_full = np.dot(diff_a_row, diff_b_row)
                    term1_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                    term1_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                    sum1 = sum1_full - term1_i - term1_j
                    
                    diff_a_col = a[:, i] - a[:, j]
                    diff_b_col = B_p[:, v] - B_p[:, u]
                    sum2_full = np.dot(diff_a_col, diff_b_col)
                    term2_i = (a[i, i] - a[i, j]) * (b[u, v] - b[u, u])
                    term2_j = (a[j, i] - a[j, j]) * (b[v, v] - b[v, u])
                    sum2 = sum2_full - term2_i - term2_j
                    
                    direct = (a[i, j] * (b[v, u] - b[u, v]) + 
                              a[j, i] * (b[u, v] - b[v, u]) + 
                              a[i, i] * (b[v, v] - b[u, u]) + 
                              a[j, j] * (b[u, u] - b[v, v]))
                    
                    deltas[i, j] = sum1 + sum2 + direct
            return deltas
    
    # Initial solution
    p = init_rank_matching()
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    no_improve_count = 0
    max_no_improve = 300
    
    while time.time() < deadline:
        B_p = b[:, p]
        
        # Compute all swap deltas
        deltas = vec_swap_delta(p, B_p)
        
        # Find best swap
        min_idx = np.unravel_index(np.argmin(deltas), deltas.shape)
        best_delta = deltas[min_idx]
        
        if best_delta >= 0:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation: random restart with k swaps
                k = min(n, 5 + random.randint(0, 10))
                for _ in range(k):
                    i = random.randint(n)
                    j = random.randint(n)
                    if i != j:
                        p[i], p[j] = p[j], p[i]
                current_cost = calc_cost(p)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                no_improve_count = 0
                continue
        else:
            # Perform the best swap
            i, j = min_idx
            p[i], p[j] = p[j], p[i]
            current_cost += int(best_delta)
            no_improve_count = 0
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
