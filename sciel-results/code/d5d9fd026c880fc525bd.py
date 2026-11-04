import time
import numpy as np
from itertools import product

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
    
    # Try to reduce rank by finding linear dependencies
    # Strategy: for each term r, check if it can be expressed as a linear combination
    # of the other terms. This means there exist coefficients c_s (s != r) such that:
    # U[:,r]*V[:,r]*W[:,r] = sum_{s!=r} c_s * U[:,s]*V[:,s]*W[:,s]
    # as tensors.
    
    # A more practical approach: try removing each term and see if the remaining
    # terms can be rescaled to still produce the correct tensor.
    
    # For each term r to remove, we need to find scalars alpha_s (s != r) such that:
    # sum_{s != r} alpha_s * U[:,s]*V[:,s]*W[:,s] = T (the target tensor)
    
    # This is a system of 48 equations (one per tensor entry) in 41 unknowns.
    # Let's set up this linear system.
    
    def try_remove_term(U_mat, V_mat, W_mat, r_to_remove, n, m, p):
        """Try to find if term r_to_remove is redundant by solving for rescaling of others."""
        rank = U_mat.shape[1]
        if rank <= 1:
            return None
        
        # Build the linear system: for each (i,j,k), 
        # sum_{s != r_to_remove} alpha_s * U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1
        
        # Number of unknowns: rank - 1
        num_unknowns = rank - 1
        # Number of equations: n*m*p
        num_eqs = n * m * p
        
        # Build matrix A and vector b
        A = np.zeros((num_eqs, num_unknowns))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    unknown_idx = 0
                    for s in range(rank):
                        if s == r_to_remove:
                            continue
                        A[eq_idx, unknown_idx] = U_mat[i*m + j, s] * V_mat[j*p + k, s] * W_mat[k*n + i, s]
                        unknown_idx += 1
                    eq_idx += 1
        
        # Solve the linear system
        try:
            # Use least squares if overdetermined
            alpha, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
            
            # Check if the solution is valid
            if residuals is None or len(residuals) == 0:
                # Check residual
                residual = np.linalg.norm(A @ alpha - b)
            else:
                residual = residuals[0]
            
            if residual < 1e-7:
                # Valid! Build the new decomposition
                keep_indices = [s for s in range(rank) if s != r_to_remove]
                U_new = U_mat[:, keep_indices].copy()
                V_new = V_mat[:, keep_indices].copy()
                W_new = W_mat[:, keep_indices].copy()
                
                # Apply the scaling
                for idx, s in enumerate(keep_indices):
                    U_new[:, idx] *= alpha[idx]
                    # V and W don't need scaling since we absorbed it into U
                    # Actually, we need to be careful: the product U*V*W must equal the target
                    # If we scale U[:,s] by alpha_s, then we need to keep V and W unchanged
                    # and the product becomes alpha_s * U[:,s]*V[:,s]*W[:,s]
                    # which is what we want.
                
                # Verify
                if verify_decomposition(U_new, V_new, W_new, n, m, p):
                    return (U_new, V_new, W_new)
                else:
                    return None
            else:
                return None
        except np.linalg.LinAlgError:
            return None
    
    # Try removing each term one at a time
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_s
    
    for remove_r in range(rank_s):
        if time.time() > deadline:
            break
        
        result = try_remove_term(U, V, W, remove_r, n, m, p)
        if result is not None:
            U, V, W = result
            rank_s = best_rank - 1
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            # Restart the loop since indices changed
            break
    
    # Try a few more rounds of reduction
    for _ in range(5):
        if time.time() > deadline:
            break
        progress = False
        for remove_r in range(rank_s):
            if time.time() > deadline:
                break
            
            result = try_remove_term(U, V, W, remove_r, n, m, p)
            if result is not None:
                U, V, W = result
                rank_s -= 1
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                progress = True
                break
        
        if not progress:
            break
    
    # Final verification
    if not verify_decomposition(U, V, W, n, m, p):
        # Fallback to the best verified result
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
