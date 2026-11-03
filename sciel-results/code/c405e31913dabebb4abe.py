import numpy as np
import time
import random

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 2.0

def _remaining():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _START_TIME)

def construct():
    n, m, p = 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def _verify(U_mat, V_mat, W_mat):
        """Check if U, V, W correctly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U_mat[0])):
                        s += U_mat[i*m+j][r] * V_mat[j*p+k][r] * W_mat[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def add_strassen(col_offset, j_offset, k_offset):
        U[0 * m + j_offset + 0][col_offset + 0] = 1
        U[1 * m + j_offset + 1][col_offset + 0] = 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 0] = 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 0] = 1
        W[(k_offset + 0) * n + 0][col_offset + 0] = 1
        W[(k_offset + 1) * n + 1][col_offset + 0] = 1

        U[1 * m + j_offset + 0][col_offset + 1] = 1
        U[1 * m + j_offset + 1][col_offset + 1] = 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 1] = 1
        W[(k_offset + 0) * n + 1][col_offset + 1] = 1
        W[(k_offset + 1) * n + 1][col_offset + 1] = -1

        U[0 * m + j_offset + 0][col_offset + 2] = 1
        V[(j_offset + 0) * p + k_offset + 1][col_offset + 2] = 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 2] = -1
        W[(k_offset + 1) * n + 0][col_offset + 2] = 1
        W[(k_offset + 1) * n + 1][col_offset + 2] = 1

        U[1 * m + j_offset + 1][col_offset + 3] = 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 3] = -1
        V[(j_offset + 1) * p + k_offset + 0][col_offset + 3] = 1
        W[(k_offset + 0) * n + 0][col_offset + 3] = 1
        W[(k_offset + 0) * n + 1][col_offset + 3] = 1

        U[0 * m + j_offset + 0][col_offset + 4] = 1
        U[0 * m + j_offset + 1][col_offset + 4] = 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 4] = 1
        W[(k_offset + 0) * n + 0][col_offset + 4] = -1
        W[(k_offset + 1) * n + 0][col_offset + 4] = 1

        U[0 * m + j_offset + 0][col_offset + 5] = -1
        U[1 * m + j_offset + 0][col_offset + 5] = 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 5] = 1
        V[(j_offset + 0) * p + k_offset + 1][col_offset + 5] = 1
        W[(k_offset + 1) * n + 1][col_offset + 5] = 1

        U[0 * m + j_offset + 1][col_offset + 6] = 1
        U[1 * m + j_offset + 1][col_offset + 6] = -1
        V[(j_offset + 1) * p + k_offset + 0][col_offset + 6] = 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 6] = 1
        W[(k_offset + 0) * n + 0][col_offset + 6] = 1

    def add_schoolbook_2x2x1(col_offset, j_offset, k_offset):
        U[0 * m + j_offset + 0][col_offset + 0] = 1
        V[(j_offset + 0) * p + k_offset][col_offset + 0] = 1
        W[k_offset * n + 0][col_offset + 0] = 1
        
        U[0 * m + j_offset + 1][col_offset + 1] = 1
        V[(j_offset + 1) * p + k_offset][col_offset + 1] = 1
        W[k_offset * n + 0][col_offset + 1] = 1
        
        U[1 * m + j_offset + 0][col_offset + 2] = 1
        V[(j_offset + 0) * p + k_offset][col_offset + 2] = 1
        W[k_offset * n + 1][col_offset + 2] = 1
        
        U[1 * m + j_offset + 1][col_offset + 3] = 1
        V[(j_offset + 1) * p + k_offset][col_offset + 3] = 1
        W[k_offset * n + 1][col_offset + 3] = 1

    col = 0
    add_strassen(col, 0, 0); col += 7
    add_strassen(col, 0, 2); col += 7
    add_schoolbook_2x2x1(col, 0, 4); col += 4
    
    add_strassen(col, 2, 0); col += 7
    add_strassen(col, 2, 2); col += 7
    add_schoolbook_2x2x1(col, 2, 4); col += 4
    
    # Verify correctness
    if not _verify(U, V, W):
        raise RuntimeError("Decomposition verification failed")
    
    # Convert to numpy for efficient operations
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Step 1: Check if any single column is redundant (simple removal)
    for r in range(rank):
        if _remaining() < 5:
            break
        U_test = np.delete(U_np, r, axis=1)
        V_test = np.delete(V_np, r, axis=1)
        W_test = np.delete(W_np, r, axis=1)
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r2 in range(rank - 1):
                        s += U_test[i*m+j, r2] * V_test[j*p+k, r2] * W_test[k*n+i, r2]
                    if abs(s - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_rank = rank - 1
            best_U = U_test.tolist()
            best_V = V_test.tolist()
            best_W = W_test.tolist()
            U_np, V_np, W_np = U_test, V_test, W_test
            rank = best_rank
            break
    
    # Step 2: Border rank perturbation - try to find linear dependencies
    # For each pair of columns (r1, r2), check if we can replace them with one column
    # by solving for a perturbation that makes the tensor equation hold to O(epsilon)
    # while setting one column to zero.
    
    # More practical approach: try all pairs and see if one can be expressed as a
    # linear combination that allows elimination
    for r1 in range(rank):
        if _remaining() < 10:
            break
        for r2 in range(r1 + 1, rank):
            if _remaining() < 10:
                break
            
            # Try removing r2 and adjusting r1
            # The contribution of term r to tensor element (i,j,k) is:
            # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # 
            # If we remove r2, we need the remaining terms to still sum to 1.
            # The deficit from removing r2 is:
            # deficit[i,j,k] = U[i*m+j, r2] * V[j*p+k, r2] * W[k*n+i, r2]
            #
            # We try to absorb this deficit by modifying term r1.
            # New contribution of r1: (U_r1 + delta_U) * (V_r1 + delta_V) * (W_r1 + delta_W)
            # To first order: U_r1*V_r1*W_r1 + delta_U*V_r1*W_r1 + U_r1*delta_V*W_r1 + U_r1*V_r1*delta_W
            # We need: delta_U*V_r1*W_r1 + U_r1*delta_V*W_r1 + U_r1*V_r1*delta_W = deficit
            #
            # This is a system of n*m*p = 40 equations with 8+8+10=26 unknowns (delta_U has 8, delta_V has 20, delta_W has 10)
            # Wait, U has n*m=8 rows, V has m*p=20 rows, W has p*n=10 rows
            # So delta_U: 8 vars, delta_V: 20 vars, delta_W: 10 vars = 38 unknowns
            # 40 equations, 38 unknowns - might be overdetermined but let's try least squares
            
            # Build the linear system
            # For each (i,j,k), the equation is:
            # sum over the three partial derivatives weighted by the other two factors
            
            # Actually, let's think more carefully. We want to find delta_U, delta_V, delta_W
            # such that for all (i,j,k):
            # (U_r1[i*m+j] + delta_U[i*m+j]) * (V_r1[j*p+k] + delta_V[j*p+k]) * (W_r1[k*n+i] + delta_W[k*n+i])
            # - U_r1[i*m+j] * V_r1[j*p+k] * W_r1[k*n+i]
            # = U_r2[i*m+j] * V_r2[j*p+k] * W_r2[k*n+i]
            
            # To first order in deltas:
            # delta_U[i*m+j]*V_r1[j*p+k]*W_r1[k*n+i] + U_r1[i*m+j]*delta_V[j*p+k]*W_r1[k*n+i] 
            # + U_r1[i*m+j]*V_r1[j*p+k]*delta_W[k*n+i]
            # = U_r2[i*m+j]*V_r2[j*p+k]*W_r2[k*n+i]
            
            # This is linear in the deltas. Let's set up the matrix.
            # Unknowns: delta_U (8), delta_V (20), delta_W (10) = 38 total
            # Equations: 40 (one for each i,j,k)
            
            A_mat = np.zeros((n*m*p, 38))
            b_vec = np.zeros(n*m*p)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a_idx = i*m + j
                        b_idx = j*p + k
                        c_idx = k*n + i
                        
                        # Coefficient for delta_U[a_idx]
                        A_mat[idx, a_idx] = V_r1[b_idx] * W_r1[c_idx]
                        
                        # Coefficient for delta_V[b_idx]
                        A_mat[idx, 8 + b_idx] = U_r1[a_idx] * W_r1[c_idx]
                        
                        # Coefficient for delta_W[c_idx]
                        A_mat[idx, 8 + 20 + c_idx] = U_r1[a_idx] * V_r1[b_idx]
                        
                        # RHS
                        b_vec[idx] = U_r2[a_idx] * V_r2[b_idx] * W_r2[c_idx]
                        
                        idx += 1
            
            # Solve least squares
            try:
                sol, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                delta_U = sol[:8]
                delta_V = sol[8:28]
                delta_W = sol[28:38]
                
                # Check if the solution is reasonable (small deltas)
                if np.max(np.abs(delta_U)) < 100 and np.max(np.abs(delta_V)) < 100 and np.max(np.abs(delta_W)) < 100:
                    # Apply the perturbation
                    U_new = U_np.copy()
                    V_new = V_np.copy()
                    W_new = W_np.copy()
                    
                    U_new[:, r1] += delta_U
                    V_new[:, r1] += delta_V
                    W_new[:, r1] += delta_W
                    U_new[:, r2] = 0
                    V_new[:, r2] = 0
                    W_new[:, r2] = 0
                    
                    # Verify
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for r3 in range(rank):
                                    s += U_new[i*m+j, r3] * V_new[j*p+k, r3] * W_new[k*n+i, r3]
                                if abs(s - 1) > 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Remove column r2
                        U_final = np.delete(U_new, r2, axis=1)
                        V_final = np.delete(V_new, r2, axis=1)
                        W_final = np.delete(W_new, r2, axis=1)
                        
                        # Final verification
                        valid_final = True
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    s = 0
                                    for r3 in range(rank - 1):
                                        s += U_final[i*m+j, r3] * V_final[j*p+k, r3] * W_final[k*n+i, r3]
                                    if abs(s - 1) > 1e-9:
                                        valid_final = False
                                        break
                                if not valid_final:
                                    break
                            if not valid_final:
                                break
                        
                        if valid_final:
                            best_rank = rank - 1
                            best_U = U_final.tolist()
                            best_V = V_final.tolist()
                            best_W = W_final.tolist()
                            return {"U": best_U, "V": best_V, "W": best_W}
            except:
                pass
    
    if best_rank < 36:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": U, "V": V, "W": W}
