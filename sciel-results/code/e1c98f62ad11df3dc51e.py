"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse import csc_matrix
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.sparse.linalg import eigsh
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    # --- Helper: compute exact QAP cost for a permutation p ---
    def compute_cost(p):
        # p[i] = location for facility i
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Efficient: precompute b_perm = b[np.ix_(p, p)]
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))

    # --- Helper: 2-opt / swap local search with incremental cost update ---
    def local_search(p, max_iters=None):
        """
        p: numpy array of int64, p[i] = location of facility i
        Returns improved permutation and cost.
        """
        p = p.copy()
        # Precompute row sums of a weighted by current placement
        # cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
        # For swap of i and k: delta = sum_j a[i,j]*(b[p[k],p[j]] - b[p[i],p[j]])
        #                         + sum_j a[k,j]*(b[p[i],p[j]] - b[p[k],p[j]])
        #                         + a[i,k]*(b[p[k],p[i]] - b[p[i],p[k]])
        #                         + a[k,i]*(b[p[i],p[k]] - b[p[k],p[i]])
        # But simpler: just recompute the two rows/cols affected.
        
        current_cost = compute_cost(p)
        n = len(p)
        
        if max_iters is None:
            max_iters = n * 20
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i in range(n):
                if time.time() > deadline:
                    break
                best_delta = 0
                best_k = -1
                pi = p[i]
                for k in range(i + 1, n):
                    pk = p[k]
                    # Compute delta for swapping i and k
                    # Old contribution involving i or k:
                    # sum_j a[i,j]*b[pi,p[j]] + sum_j a[k,j]*b[pk,p[j]]
                    # + sum_j a[j,i]*b[p[j],pi] + sum_j a[j,k]*b[p[j],pk]
                    # But since a and b are symmetric in QAPLIB (typically), 
                    # we can simplify. Let's just compute directly for correctness.
                    # Actually, let's use the efficient formula:
                    # delta = sum_j (a[i,j] - a[k,j]) * (b[pk,p[j]] - b[pi,p[j]])
                    #       + a[i,k]*(b[pk,pi] - b[pi,pk]) + a[k,i]*(b[pi,pk] - b[pk,pi])
                    # For symmetric a and b: a[i,k]=a[k,i], b[pi,pk]=b[pk,pi]
                    # delta = (a[i,k] - a[k,i]) * (b[pk,pi] - b[pi,pk]) ... 
                    # Let's just do the direct but efficient computation:
                    
                    # New cost if we swap i and k:
                    # The terms that change are those involving i or k.
                    # Old: row_i + row_k + col_i + col_k (but avoid double counting i,k)
                    # Let's compute:
                    # old_contrib = sum_j a[i,j]*b[pi,p[j]] + sum_j a[k,j]*b[pk,p[j]]
                    #            + sum_j a[j,i]*b[p[j],pi] + sum_j a[j,k]*b[p[j],pk]
                    #            - a[i,k]*b[pi,pk] - a[k,i]*b[pk,pi]  # subtract double-counted
                    # new_contrib = same but with pi and pk swapped in b, and p[i],p[k] swapped
                    # This is getting complex. Let's just recompute cost for the swap
                    # by updating only the affected terms.
                    
                    # Simpler approach: compute the change directly
                    # delta = sum_j a[i,j]*(b[pk,p[j]] - b[pi,p[j]])
                    #       + sum_j a[k,j]*(b[pi,p[j]] - b[pk,p[j]])
                    #       + sum_j a[j,i]*(b[p[j],pk] - b[p[j],pi])
                    #       + sum_j a[j,k]*(b[p[j],pi] - b[p[j],pk])
                    #       - a[i,k]*b[pi,pk] - a[k,i]*b[pk,pi]
                    #       + a[i,k]*b[pk,pi] + a[k,i]*b[pi,pk]
                    #       - a[i,k]*b[pk,pi] - a[k,i]*b[pi,pk]
                    #       + a[i,k]*b[pi,pk] + a[k,i]*b[pk,pi]
                    
                    # Actually, the cleanest way:
                    # delta = sum_j (a[i,j] - a[k,j]) * (b[pk,p[j]] - b[pi,p[j]])
                    #       + sum_j (a[j,i] - a[j,k]) * (b[p[j],pk] - b[p[j],pi])
                    #       + a[i,k]*(b[pk,pi] - b[pi,pk])
                    #       + a[k,i]*(b[pi,pk] - b[pk,pi])
                    #       - a[i,k]*b[pi,pk] - a[k,i]*b[pk,pi]
                    #       + a[i,k]*b[pk,pi] + a[k,i]*b[pi,pk]
                    #       - a[i,k]*b[pk,pi] - a[k,i]*b[pi,pk]
                    #       + a[i,k]*b[pi,pk] + a[k,i]*b[pk,pi]
                    
                    # For symmetric matrices (which wil100 has), this simplifies to:
                    # delta = 2 * sum_j (a[i,j] - a[k,j]) * (b[pk,p[j]] - b[pi,p[j]])
                    #       - 2 * a[i,k] * (b[pi,pk] - b[pk,pi])  # which is 0 if b symmetric
                    # So delta = 2 * sum_j (a[i,j] - a[k,j]) * (b[pk,p[j]] - b[pi,p[j]])
                    
                    # But to be safe, let's compute it properly:
                    diff_a = a[i, :] - a[k, :]
                    diff_b_row = b[pk, :] - b[pi, :]
                    term1 = np.dot(diff_a, diff_b_row)
                    
                    diff_a_col = a[:, i] - a[:, k]
                    diff_b_col = b[:, pk] - b[:, pi]
                    term2 = np.dot(diff_a_col, diff_b_col)
                    
                    # Diagonal correction: we double-counted (i,k) and (k,i)
                    # term1 includes j=k: a[i,k]-a[k,k] times b[pk,pk]-b[pi,pk]
                    # term2 includes j=k: a[k,k]-a[k,k] times b[pk,pk]-b[pi,pk] = 0
                    # term1 includes j=i: a[i,i]-a[k,i] times b[pk,pi]-b[pi,pi]
                    # term2 includes j=i: a[i,i]-a[k,i] times b[pi,pk]-b[pi,pk] = 0
                    # The cross terms a[i,k]*b[pi,pk] and a[k,i]*b[pk,pi] are handled
                    # by the fact that in the original sum, a[i,k]*b[pi,pk] appears once
                    # and after swap it becomes a[i,k]*b[pk,pi].
                    # For symmetric b, b[pi,pk]=b[pk,pi], so no change from these.
                    # For non-symmetric, we need:
                    # delta_cross = a[i,k]*(b[pk,pi]-b[pi,pk]) + a[k,i]*(b[pi,pk]-b[pk,pi])
                    #              = (a[i,k]-a[k,i])*(b[pk,pi]-b[pi,pk])
                    delta = term1 + term2 + (a[i,k] - a[k,i]) * (b[pk,pi] - b[pi,pk])
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_k = k
                if best_k >= 0:
                    p[i], p[best_k] = p[best_k], p[i]
                    current_cost += best_delta
                    improved = True
            if time.time() > deadline:
                break
        return p, current_cost

    # --- Strategy 1: Spectral embedding initialization ---
    def spectral_init():
        """
        Use Laplacian eigenvectors to embed facility and location graphs,
        then match via Hungarian algorithm on the distance matrix.
        """
        # Compute Laplacian of facility graph (matrix a)
        # L_a = D_a - a, where D_a = diag(a.sum(axis=1))
        a_sym = (a + a.T) / 2.0  # ensure symmetry for Laplacian
        b_sym = (b + b.T) / 2.0
        
        # Use sparse Laplacian for efficiency
        a_sp = csc_matrix(a_sym)
        b_sp = csc_matrix(b_sym)
        
        L_a = sparse_laplacian(a_sp, normed=False)
        L_b = sparse_laplacian(b_sp, normed=False)
        
        # Get smallest non-trivial eigenvectors
        # We want the first k eigenvectors (excluding the trivial zero eigenvalue)
        k = min(10, n - 1)
        
        try:
            eigvals_a, eigvecs_a = eigsh(L_a, k=k+1, which='SM')
            # Sort by eigenvalue
            idx = np.argsort(eigvals_a)
            eigvecs_a = eigvecs_a[:, idx]
            # Take the first k (skip the smallest which is ~0)
            X_a = eigvecs_a[:, 1:k+1]
        except Exception:
            X_a = np.random.randn(n, k)
        
        try:
            eigvals_b, eigvecs_b = eigsh(L_b, k=k+1, which='SM')
            idx = np.argsort(eigvals_b)
            eigvecs_b = eigvecs_b[:, idx]
            X_b = eigvecs_b[:, 1:k+1]
        except Exception:
            X_b = np.random.randn(n, k)
        
        # Compute distance matrix between embeddings
        # dist[i,j] = ||X_a[i] - X_b[j]||^2
        # = ||X_a[i]||^2 + ||X_b[j]||^2 - 2 * X_a[i] @ X_b[j]^T
        norm_a = np.sum(X_a**2, axis=1)
        norm_b = np.sum(X_b**2, axis=1)
        cross = X_a @ X_b.T
        dist = norm_a[:, None] + norm_b[None, :] - 2 * cross
        
        # Hungarian algorithm to minimize total distance
        row_ind, col_ind = linear_sum_assignment(dist)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    # --- Strategy 2: Rank-matching initialization (from seed) ---
    def rank_init():
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    # --- Strategy 3: Random initialization ---
    def random_init():
        return np.random.permutation(n).astype(np.int64)

    # Generate candidate initial solutions
    candidates = []
    
    # Spectral
    try:
        p_spec = spectral_init()
        candidates.append(p_spec)
    except Exception:
        pass
    
    # Rank
    p_rank = rank_init()
    candidates.append(p_rank)
    
    # A few random
    for _ in range(3):
        candidates.append(random_init())

    # Local search on each candidate
    best_p = None
    best_cost = np.inf
    
    for p_init in candidates:
        if time.time() > deadline - 10:
            break
        p_ls, cost_ls = local_search(p_init)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Final check
    if best_p is None:
        best_p = candidates[0]
        best_cost = compute_cost(best_p)
    
    # One more pass of local search if time permits
    if time.time() < deadline - 5:
        best_p, best_cost = local_search(best_p)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
