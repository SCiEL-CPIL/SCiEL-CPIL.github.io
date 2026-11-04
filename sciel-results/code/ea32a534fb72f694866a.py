"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    
    # 1. Prepare MDS Embeddings
    # b is distance matrix. Squared distances for MDS.
    D_sq = b ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    M = -0.5 * J @ D_sq @ J
    M = (M + M.T) / 2
    
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Ensure non-negative
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # Generate diverse embeddings using random rotations of the top eigenvectors
    # This creates different "geometric" views of the problem
    num_embeddings = 10
    candidates = []
    
    # Pre-compute for local search
    a_float = a.astype(np.float64)
    b_float = b.astype(np.float64)
    
    start_time = time.time()
    time_limit = 110
    
    for e in range(num_embeddings):
        if time.time() - start_time > time_limit - 20:
            break
            
        # Create a random linear combination of the top eigenvectors
        # This simulates a different "force field" or embedding
        if e == 0:
            Y = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        else:
            # Random rotation in the subspace spanned by top eigenvectors
            # Or just pick a random subset of dimensions
            # Let's use a random projection
            dim = min(n, 20 + 5 * e)
            if dim > len(eigenvalues):
                dim = len(eigenvalues)
            
            # Random matrix R of shape (n, dim)
            R = np.random.randn(n, dim)
            # Project eigenvectors onto R? Or just use random linear combos of eigenvectors
            # Y = eigenvectors[:, :dim] @ np.random.randn(dim, dim)
            # Let's stick to using the principal components but with random signs/rotations to ensure diversity
            # Actually, just using different numbers of dimensions is a good start.
            # To be more diverse, we can add a small random perturbation to the embedding
            Y = eigenvectors[:, :dim] @ np.diag(np.sqrt(eigenvalues[:dim]))
            
            # Add small random noise to break symmetries
            Y += 0.1 * np.random.randn(n, dim)
            
        # Symplectic Dynamics on Stiefel Manifold
        # We want to maximize tr(A X G X^T) where G = Y Y^T
        # Potential E = -tr(A X G X^T)
        # Force = 2 A X G
        
        G = Y.T @ Y
        X = Y.copy()
        V = np.zeros_like(X)
        
        dt = 0.05
        num_steps = 50
        sample_freq = 10
        
        local_candidates = []
        
        for step in range(num_steps):
            F = 2 * (a_float @ X) @ G
            
            X_half = X + 0.5 * dt * V
            # Project to Stiefel
            U, s, Vt = np.linalg.svd(X_half, full_matrices=False)
            X_half = U @ Vt
            
            F_half = 2 * (a_float @ X_half) @ G
            
            V = V + 0.5 * dt * (F + F_half)
            
            X = X_half + 0.5 * dt * V
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            X = U @ Vt
            
            if step % sample_freq == 0:
                # Discretize
                # Cost matrix for assignment: C[i, j] = -tr(a[i, i] b[p[i], p[j]])? 
                # Approximation: assign facility i to location j minimizing distance to "center of mass" of high-flow neighbors?
                # Better: Use the linear assignment approximation.
                # Cost[i, j] = sum_k a[i, k] b[j, p[k]] is not static.
                # Standard heuristic: Linearize the quadratic term.
                # Let's use the current X to define a cost.
                # C[i, j] = -2 * (a_float @ X @ G @ X.T)[i, j]? No.
                
                # Simple heuristic: Assign based on similarity in embedding space
                # High flow pairs should be close.
                # We can compute a "weighted distance" for each facility i to each location j.
                # dist[i, j] = sum_k a[i, k] b[j, p[k]]
                # This requires a current permutation p.
                # Let's initialize p with a simple heuristic (e.g., matching centers)
                
                # For the first sample, we need a p.
                # Let's use the Hungarian algorithm on a cost matrix derived from X and Y.
                # Cost[i, j] = || X[i] - Y[j] ||^2
                C = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2 * (X @ Y.T)
                row_ind, col_ind = linear_sum_assignment(C)
                p = col_ind.copy()
                
                # Refine p with a few steps of 2-opt to get a good candidate
                p = fast_local_search(a_float, b_float, p, time_limit, start_time)
                
                cost = np.sum(a_float * b_float[p[:, None], p[None, :]])
                local_candidates.append((cost, p))
        
        candidates.extend(local_candidates)
        
    # Sort candidates
    candidates.sort(key=lambda x: x[0])
    
    best_p = None
    best_cost = float('inf')
    
    if candidates:
        best_cost, best_p = candidates[0]
        # Run one more round of local search on the best to ensure convergence
        if time.time() - start_time < time_limit - 5:
            best_p = fast_local_search(a_float, b_float, best_p, time_limit, start_time)
            best_cost = np.sum(a_float * b_float[best_p[:, None], best_p[None, :]])
            
    if best_p is None:
        # Fallback
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        best_p = np.empty(n, dtype=np.int64)
        best_p[heavy] = central
        
    return best_p.tolist()


def fast_local_search(a, b, p, time_limit, start_time):
    n = len(p)
    p = p.copy()
    current_cost = np.sum(a * b[p[:, None], p[None, :]])
    
    # Precompute row/col sums for quick delta calculation?
    # We will use the vectorized delta calculation.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit - 5:
        improved = False
        
        # Precompute b[p, :] and b[:, p] for current permutation
        # b_p = b[p, :] is not quite right, we need b[p[i], p[k]]
        # Let's just compute the deltas on the fly, but vectorized over j for each i.
        
        for i in range(n):
            if (time.time() - start_time) >= time_limit - 5:
                break
                
            u = p[i]
            # We want to find j > i that minimizes delta
            # Delta(i, j) = (a[i, :] - a[j, :]) dot (b[v, p] - b[u, p]) + (a[:, i] - a[:, j]) dot (b[p, v] - b[p, u]) + direct
            
            # To vectorize over j, we need b[v, p] for all v.
            # b[v, p] is a matrix B_p where B_p[v, k] = b[v, p[k]].
            # This is just b[:, p] transposed? No.
            # b[:, p] has shape (n, n). b[:, p][v, k] = b[v, p[k]]. Yes.
            B_col = b[:, p] # Shape (n, n). B_col[v, k] = b[v, p[k]]
            B_row = b[p, :] # Shape (n, n). B_row[k, v] = b[p[k], v]
            
            # For a fixed i (loc u), we want to compare with all j (loc v)
            # Term 1: (a[i, :] - a[j, :]) dot (B_col[v, :] - B_col[u, :])
            # Let d_u = B_col[u, :]
            # Let d_v = B_col[v, :]
            # Term 1 = (a[i, :] - a[j, :]) dot (d_v - d_u)
            # = a[i, :] dot d_v - a[i, :] dot d_u - a[j, :] dot d_v + a[j, :] dot d_u
            
            # We can precompute:
            # A_i_dot_d = a[i, :] dot B_col.T ? No.
            # a[i, :] dot d_v = a[i, :] dot B_col[v, :]
            # This is the v-th element of (a[i, :] @ B_col.T)? No.
            # (a[i, :] @ B_col.T)[v] = sum_k a[i, k] B_col.T[v, k] = sum_k a[i, k] B_col[k, v] = sum_k a[i, k] b[p[k], v].
            # This is the "column" contribution of facility i if it were at v?
            
            # Let's compute:
            # S1[v] = sum_k a[i, k] b[p[k], v]  -> This is (a[i, :] @ B_row)[v]? 
            # B_row[k, v] = b[p[k], v]. Yes.
            # So S1 = a[i, :] @ B_row
            
            # S2[v] = sum_k a[k, i] b[p[k], v]
            # This is (a[:, i] @ B_row)[v]? No.
            # B_row is b[p, :].
            # We need sum_k a[k, i] b[p[k], v].
            # Let A_col_i = a[:, i].
            # This is A_col_i @ B_row? No, A_col_i is (n,). B_row is (n, n).
            # (A_col_i @ B_row)[v] = sum_k a[k, i] b[p[k], v]. Yes.
            
            # So for fixed i:
            # S1 = a[i, :] @ B_row  (vector of length n)
            # S2 = a[:, i] @ B_row  (vector of length n)
            
            # Now for the swap with j (loc v):
            # Change in "row" terms for i and j:
            # i moves u -> v: S1[v] - S1[u]
            # j moves v -> u: S1_j[u] - S1_j[v] where S1_j = a[j, :] @ B_row
            
            # This is getting O(n^2) per i if we compute S1_j for all j.
            # S1_all = a @ B_row. Shape (n, n).
            # S1_all[i, v] = sum_k a[i, k] b[p[k], v].
            
            # Similarly for column terms:
            # S3[v] = sum_k a[k, i] b[v, p[k]]
            # B_col[v, k] = b[v, p[k]].
            # S3 = a[:, i] @ B_col?
            # (a[:, i] @ B_col)[v] = sum_k a[k, i] b[v, p[k]]. Yes.
            # S3_all = a.T @ B_col. Shape (n, n).
            # S3_all[i, v] = sum_k a[k, i] b[v, p[k]].
            
            S1_all = a @ B_row
            S3_all = a.T @ B_col
            
            # For fixed i, u=p[i]:
            # Cost change for moving i to v (ignoring j for a moment) is:
            # Delta_i_to_v = S1_all[i, v] - S1_all[i, u] + S3_all[i, v] - S3_all[i, u] + a[i, i] * (b[v, v] - b[u, u])
            
            # For swap i, j (u, v):
            # Delta = (S1_all[i, v] - S1_all[i, u]) + (S3_all[i, v] - S3_all[i, u])
            #       + (S1_all[j, u] - S1_all[j, v]) + (S3_all[j, u] - S3_all[j, v])
            #       + a[i, j] * (b[v, u] - b[u, v]) + a[j, i] * (b[u, v] - b[v, u])
            #       + a[i, i] * (b[v, v] - b[u, u]) + a[j, j] * (b[u, u] - b[v, v])
            
            # Since b is symmetric, b[v, u] = b[u, v].
            # Cross terms: a[i, j] * 0 + a[j, i] * 0 = 0.
            # Self terms: b[v, v] = 0, b[u, u] = 0. So 0.
            
            # So Delta = (S1_all[i, v] - S1_all[i, u]) + (S3_all[i, v] - S3_all[i, u])
            #           + (S1_all[j, u] - S1_all[j, v]) + (S3_all[j, u] - S3_all[j, v])
            
            # Let D_i[v] = S1_all[i, v] + S3_all[i, v]
            # Let D_j[u] = S1_all[j, u] + S3_all[j, u]
            # Delta = D_i[v] - D_i[u] + D_j[u] - D_j[v]
            
            D_all = S1_all + S3_all
            
            # For fixed i, u:
            # Delta(j) = D_i[v] - D_i[u] + D_j[u] - D_j[v]
            # = (D_i[v] - D_j[v]) - (D_i[u] - D_j[u])
            
            # We want to minimize Delta(j) for j != i.
            # Let M[v] = D_i[v] - D_j[v]. This depends on j.
            # This form is hard to vectorize over j directly because D_j depends on j.
            
            # Let's rewrite:
            # Delta(j) = D_i[v] - D_i[u] + D_j[u] - D_j[v]
            # = (D_i[v] - D_j[v]) - (D_i[u] - D_j[u])
            
            # We can compute this for all j in a vectorized way?
            # D_i is a vector. D_all is a matrix.
            # D_j is D_all[j, :].
            # We want to find j that minimizes:
            # D_i[p[j]] - D_all[j, p[j]] - (D_i[u] - D_all[j, u])
            
            # Let's just loop over j but use precomputed D_i and D_all.
            # This is O(n) per i, so O(n^2) total.
            
            D_i = D_all[i, :]
            best_delta = 0
            best_j = -1
            
            # We need D_all[j, u] and D_all[j, p[j]]
            # p is the permutation.
            # D_all[j, u] is a column of D_all.
            # D_all[j, p[j]] is a diagonal-like element.
            
            # Let's compute the vector of deltas for all j
            # delta_j = D_i[p[j]] - D_all[j, p[j]] - (D_i[u] - D_all[j, u])
            
            # D_i[p[j]]: gather D_i at indices p
            # D_all[j, p[j]]: gather diagonal of D_all[p, p]? No.
            # D_all[j, p[j]] is the element at row j, col p[j].
            
            # Let's create a matrix D_p = D_all[:, p]
            # D_p[j, k] = D_all[j, p[k]]
            # We want D_p[j, j] (since p[j] is the location of j, so k=j? No, p[k] is location of k.
            # We want D_all[j, p[j]]. This is the element in row j, column p[j] of D_all.
            
            # Let's just do the loop, it's fast enough for n=100.
            for j in range(n):
                if j == i:
                    continue
                v = p[j]
                # delta = D_i[v] - D_i[u] + D_all[j, u] - D_all[j, v]
                delta = D_i[v] - D_i[u] + D_all[j, u] - D_all[j, v]
                if delta < best_delta:
                    best_delta = delta
                    best_j = j
            
            if best_j != -1 and best_delta < 0:
                p[i], p[best_j] = p[best_j], p[i]
                current_cost += best_delta
                improved = True
                
    return p
# EVOLVE-BLOCK-END
