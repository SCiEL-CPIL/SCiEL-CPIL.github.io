import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Build the target tensor: T[i,j,k] = 1 for all valid indices
    # C[k*n+i] = C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 (coefficient of A[i,j]*B[j,k] in C[i,k])
    
    # Strassen-based rank-42 decomposition
    rank_s = 42
    U = np.zeros((n * m, rank_s), dtype=float)
    V = np.zeros((m * p, rank_s), dtype=float)
    W = np.zeros((p * n, rank_s), dtype=float)
    
    def add_strassen_block(i_start, j_start, k_start, offset):
        col = offset
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1

        col = offset + 1
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] -= 1

        col = offset + 2
        U[(i_start+0)*m + (j_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1

        col = offset + 3
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] -= 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1

        col = offset + 4
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1

        col = offset + 5
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+0)][col] -= 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1

        col = offset + 6
        U[(i_start+0)*m + (j_start+1)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] -= 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1

    add_strassen_block(0, 0, 0, 0)
    add_strassen_block(0, 0, 2, 7)
    add_strassen_block(0, 0, 4, 14)
    add_strassen_block(0, 2, 0, 21)
    add_strassen_block(0, 2, 2, 28)
    add_strassen_block(0, 2, 4, 35)
    
    # Verify the Strassen solution
    def verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        rank = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U_mat[i*m + j, r] * V_mat[j*p + k, r] * W_mat[k*n + i, r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_decomposition(U, V, W, n, m, p):
        # Fallback to trivial rank-48
        rank_t = n * m * p
        U_t = np.zeros((n * m, rank_t))
        V_t = np.zeros((m * p, rank_t))
        W_t = np.zeros((p * n, rank_t))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_t[i*m + j, r] = 1
                    V_t[j*p + k, r] = 1
                    W_t[k*n + i, r] = 1
                    r += 1
        return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
    
    # Attempt to reduce rank by finding linear dependencies
    # We construct the matrix of tensor terms
    # Each term r corresponds to a vector T_r in R^(n*m*p)
    # T_r[i, j, k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    def get_tensor_terms(U_mat, V_mat, W_mat, n, m, p):
        rank = U_mat.shape[1]
        num_terms = n * m * p
        terms = np.zeros((num_terms, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        terms[idx, r] = U_mat[i*m + j, r] * V_mat[j*p + k, r] * W_mat[k*n + i, r]
                    idx += 1
        return terms
    
    def try_reduce_rank(U_mat, V_mat, W_mat, n, m, p):
        rank = U_mat.shape[1]
        terms = get_tensor_terms(U_mat, V_mat, W_mat, n, m, p)
        
        # Find linear dependencies
        # We want to find if any column is in the span of the others
        # This is equivalent to checking if the rank of the matrix is less than the number of columns
        # However, we need to maintain the decomposable form.
        
        # Strategy: Iterate through columns, try to remove one, and solve for the rest.
        # If we remove column r, we need to find coefficients alpha_s such that
        # sum_{s != r} alpha_s * T_s = T_r
        # Then we can replace T_r with this sum? No, we just remove T_r and scale the others?
        # No, the target is the sum of all terms.
        # If T_r = sum_{s != r} c_s T_s, then sum_{all} T_i = sum_{s != r} (1 + c_s) T_s.
        # So we can remove T_r and multiply T_s by (1 + c_s).
        
        best_U, best_V, best_W = U_mat.copy(), V_mat.copy(), W_mat.copy()
        current_rank = rank
        
        for r_to_remove in range(current_rank):
            if time.time() > deadline:
                break
            
            # Build system for removing r_to_remove
            # Target: T_r = sum_{s != r} c_s T_s
            # This is a linear system A c = b
            # A is (n*m*p) x (rank-1)
            # b is T_r
            
            num_eqs = n * m * p
            num_unknowns = current_rank - 1
            
            A = np.zeros((num_eqs, num_unknowns))
            b = terms[:, r_to_remove]
            
            unknown_idx = 0
            for s in range(current_rank):
                if s == r_to_remove:
                    continue
                A[:, unknown_idx] = terms[:, s]
                unknown_idx += 1
            
            try:
                c, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
                
                if residuals is not None and len(residuals) > 0:
                    residual = residuals[0]
                else:
                    residual = np.linalg.norm(A @ c - b)
                
                if residual < 1e-7:
                    # We can remove r_to_remove
                    # New coefficients for remaining terms: 1 + c_s
                    # Note: The sum is sum T_i. If T_r = sum c_s T_s, then sum T_i = sum_{s!=r} (1 + c_s) T_s.
                    # So we scale the remaining terms by (1 + c_s).
                    
                    keep_indices = [s for s in range(current_rank) if s != r_to_remove]
                    U_new = U_mat[:, keep_indices].copy()
                    V_new = V_mat[:, keep_indices].copy()
                    W_new = W_mat[:, keep_indices].copy()
                    
                    for idx, s in enumerate(keep_indices):
                        scale = 1.0 + c[idx]
                        U_new[:, idx] *= scale
                    
                    if verify_decomposition(U_new, V_new, W_new, n, m, p):
                        best_U, best_V, best_W = U_new, V_new, W_new
                        current_rank -= 1
                        # Update terms for next iteration?
                        # For simplicity, we restart the loop with the new matrices
                        # But to save time, we just break and restart the whole reduction process
                        return try_reduce_rank(best_U, best_V, best_W, n, m, p)
            except np.linalg.LinAlgError:
                pass
        
        return best_U, best_V, best_W
    
    # Run reduction
    U, V, W = try_reduce_rank(U, V, W, n, m, p)
    
    # Final verification
    if not verify_decomposition(U, V, W, n, m, p):
        # Fallback to the original Strassen solution
        U = np.zeros((n * m, 42), dtype=float)
        V = np.zeros((m * p, 42), dtype=float)
        W = np.zeros((p * n, 42), dtype=float)
        add_strassen_block(0, 0, 0, 0)
        add_strassen_block(0, 0, 2, 7)
        add_strassen_block(0, 0, 4, 14)
        add_strassen_block(0, 2, 0, 21)
        add_strassen_block(0, 2, 2, 28)
        add_strassen_block(0, 2, 4, 35)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
