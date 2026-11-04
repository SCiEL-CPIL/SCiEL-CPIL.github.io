import numpy as np
import time

def solve(a, b):
    n = a.shape[0]
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s limit

    # --- Phase 1: MDS/Procrustes Initialization ---
    # Embed facilities and locations in a low-dimensional space
    # such that the QAP objective is approximated by a Procrustes problem.
    
    # Compute distance matrices
    # For QAP, we want to minimize sum a[i,j]*b[p[i],p[j]]
    # This is similar to minimizing sum of squared distances after alignment
    
    # Use the matrices themselves as "distance" proxies
    # MDS: embed so that inner products approximate a and b
    
    def _mds_embed(matrix, dim=10):
        """Multidimensional scaling to embed matrix into dim dimensions."""
        # Center the matrix
        D = matrix.copy().astype(np.float64)
        n = D.shape[0]
        
        # Double centering
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (D ** 2) @ J
        
        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(B)
        # Sort in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Take top dim components
        eigvals = eigvals[:dim]
        eigvecs = eigvecs[:, :dim]
        
        # Ensure non-negative eigenvalues
        eigvals = np.maximum(eigvals, 0)
        
        X = eigvecs * np.sqrt(eigvals)
        return X
    
    # Embed facilities and locations
    X_fac = _mds_embed(a, dim=10)
    X_loc = _mds_embed(b, dim=10)
    
    # Procrustes alignment: find permutation that best aligns X_fac to X_loc
    # This is equivalent to finding p that minimizes ||X_fac - X_loc[:, p]||^2
    # Which expands to: -2 * sum_i X_fac[i] . X_loc[p[i]] + constants
    
    # For exact Procrustes with permutation, we can use the Hungarian algorithm
    # on the cost matrix C[i, j] = -2 * X_fac[i] . X_loc[j]
    
    from scipy.optimize import linear_sum_assignment
    
    cost_matrix = -2 * X_fac @ X_loc.T
    
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.zeros(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    # --- Phase 2: Local Search with O(1) Delta Calculation ---
    
    def compute_precomp(p):
        """Compute R and C matrices for fast delta evaluation."""
        B_col_p = b[:, p]
        B_row_p = b[p, :]
        R = a @ B_col_p.T
        C = a.T @ B_row_p
        return R, C
    
    R, C = compute_precomp(p)
    current_cost = np.sum(a * b[np.ix_(p, p)])
    
    # Local search: try all pairs, accept if improvement
    improved = True
    no_improve_count = 0
    max_no_improve = 10000
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try all pairs in a fixed order (deterministic for reproducibility)
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            for k in range(i + 1, n):
                j = p[i]
                l = p[k]
                
                if j == l:
                    continue
                
                # Calculate delta for swapping p[i] and p[k]
                # Delta = change in objective when swapping locations j and l for facilities i and k
                
                # Self terms
                d_self = (a[i, k] * b[l, j] + a[k, i] * b[j, l] + a[i, i] * b[l, l] + a[k, k] * b[j, j]) - \
                         (a[i, k] * b[j, l] + a[k, i] * b[l, j] + a[i, i] * b[j, j] + a[k, k] * b[l, l])
                
                # Cross terms with other facilities
                term_Ri = (R[i, l] - R[i, j]) - a[i, i] * (b[l, j] - b[j, j]) - a[i, k] * (b[l, l] - b[j, l])
                term_Rk = (R[k, j] - R[k, l]) - a[k, k] * (b[j, j] - b[l, l]) - a[k, i] * (b[j, l] - b[l, j])
                
                term_Ci = (C[i, l] - C[i, j]) - a[i, i] * (b[j, l] - b[j, j]) - a[k, i] * (b[l, l] - b[l, j])
                term_Ck = (C[k, j] - C[k, l]) - a[k, k] * (b[j, j] - b[l, l]) - a[i, k] * (b[j, l] - b[l, j])
                
                d_v = term_Ri + term_Rk + term_Ci + term_Ck
                delta = d_self + d_v
                
                if delta < 0:
                    # Accept swap
                    p[i], p[k] = p[k], p[i]
                    current_cost += delta
                    improved = True
                    no_improve_count = 0
                    
                    # Update R and C
                    R += np.outer(a[:, i] - a[:, k], b[:, l] - b[:, j])
                    C += np.outer(a[i, :] - a[k, :], b[l, :] - b[j, :])
                    
                    # After swap, break inner loop to restart scan
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    return p.tolist()
