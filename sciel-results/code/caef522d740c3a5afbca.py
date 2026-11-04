import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target: M_{3,3,7}
    # Strategy: Decompose into M_{3,3,3} (rank R1) and M_{3,3,4} (rank R2)
    # R1 + R2 < 63
    
    # Step 1: Find a low rank decomposition for 3x3x3
    # Try ranks 26, 25, 24, 23
    best_decomp_333 = None
    best_rank_333 = 27 # Schoolbook
    
    start_time = time.time()
    deadline = 110 # 10s margin
    
    for R in [26, 25, 24, 23]:
        if time.time() > deadline:
            break
            
        # Initialize
        U = np.random.randn(9, R)
        V = np.random.randn(9, R)
        W = np.random.randn(9, R)
        
        # Target Tensor T (flattened)
        # T[i, j, k] = 1
        T = np.ones((3, 3, 3))
        
        # Alternating Minimization
        # We want sum_r U[i,j,r] V[j,k,r] W[i,k,r] = T[i,j,k]
        # Note: W index in prompt is k*n+i. For 3x3x3, n=3.
        # So W[k*3+i, r].
        
        for _ in range(1000):
            if time.time() > deadline:
                break
                
            # Update W: For each r, W[:, r] is determined by U[:, r] and V[:, r]
            # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            # For fixed U, V, this is linear in W.
            # Let A_r = U[:, r] (vector of 9), B_r = V[:, r] (vector of 9).
            # We need sum_r (A_r * B_r) * W_r = T (reshaped)
            # Actually, it's a tensor product.
            # We can solve for W[:, r] sequentially? No, they are coupled in the sum.
            # We can use least squares for the whole W matrix.
            # Flatten T to vector t (27).
            # Construct matrix M_W such that M_W @ W_flat = t.
            # M_W has 27 rows, 9*R columns.
            # Row (i,j,k) corresponds to sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            # This is not linear in W_flat if U,V are fixed?
            # Yes it is. W appears linearly.
            
            # Build M_W
            M_W = np.zeros((27, 9*R))
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        idx_t = i*9 + j*3 + k # T index
                        for r in range(R):
                            idx_w = (k*3 + i)*R + r # W flat index
                            coeff = U[i*3+j, r] * V[j*3+k, r]
                            M_W[idx_t, idx_w] = coeff
            
            # Solve M_W @ W_flat = t
            t_flat = T.flatten()
            W_flat, residuals, rank, sv = np.linalg.lstsq(M_W, t_flat, rcond=None)
            W = W_flat.reshape((9, R))
            
            # Similarly for V
            M_V = np.zeros((27, 9*R))
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        idx_t = i*9 + j*3 + k
                        for r in range(R):
                            idx_v = (j*3+k)*R + r
                            coeff = U[i*3+j, r] * W[k*3+i, r]
                            M_V[idx_t, idx_v] = coeff
            V_flat, _, _, _ = np.linalg.lstsq(M_V, t_flat, rcond=None)
            V = V_flat.reshape((9, R))
            
            # Similarly for U
            M_U = np.zeros((27, 9*R))
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        idx_t = i*9 + j*3 + k
                        for r in range(R):
                            idx_u = (i*3+j)*R + r
                            coeff = V[j*3+k, r] * W[k*3+i, r]
                            M_U[idx_t, idx_u] = coeff
            U_flat, _, _, _ = np.linalg.lstsq(M_U, t_flat, rcond=None)
            U = U_flat.reshape((9, R))
            
        # Check error
        # Reconstruct T
        T_recon = np.zeros((3,3,3))
        for r in range(R):
            # U[:, r] is 9x1, V[:, r] is 9x1, W[:, r] is 9x1
            # Need to compute sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # This is slow in Python loops. Use numpy.
            pass # Skip full check in loop for speed, check at end
            
        # Final Check and Rounding
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify rounded
        T_check = np.zeros((3,3,3))
        for r in range(R):
            Ur = U_rounded[:, r]
            Vr = V_rounded[:, r]
            Wr = W_rounded[:, r]
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        T_check[i,j,k] += Ur[i*3+j] * Vr[j*3+k] * Wr[k*3+i]
                        
        if np.allclose(T_check, T, atol=1e-9):
            best_decomp_333 = (U_rounded, V_rounded, W_rounded)
            best_rank_333 = R
            break # Found good one, stop searching lower ranks? No, try lower? 
            # Actually, if R=26 works, 25 might work too. But 26 is enough to beat 63.
            # Let's break to save time.
            
    # Step 2: Build 3x3x7
    # Use best_decomp_333 for first 3 cols of B
    # Use Schoolbook for next 4 cols of B (rank 36)
    
    if best_decomp_333:
        U_333, V_333, W_333 = best_decomp_333
        R_333 = best_rank_333
    else:
        # Schoolbook 3x3x3
        R_333 = 27
        U_333 = np.zeros((9, R_333))
        V_333 = np.zeros((9, R_333))
        W_333 = np.zeros((9, R_333))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_333[i*3+j, col] = 1
                    V_333[j*3+k, col] = 1
                    W_333[k*3+i, col] = 1
                    col += 1
                    
    # Schoolbook for 3x3x4
    R_334 = 36
    U_334 = np.zeros((9, R_334))
    V_334 = np.zeros((9, R_334))
    W_334 = np.zeros((9, R_334))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(4): # 4 columns
                U_334[i*3+j, col] = 1
                V_334[j*3+k, col] = 1
                # W index for 3x3x4: k*n+i. n=3.
                # But wait, the full problem is 3x3x7.
                # The W matrix for the whole problem has shape (p*n, R) = (21, R).
                # The W for the 3x3x3 part corresponds to k=0,1,2.
                # The W for the 3x3x4 part corresponds to k=3,4,5,6.
                # So W_334 should be mapped to the correct rows in the final W.
                W_334[(k+3)*3+i, col] = 1 # Offset by 3
                col += 1
                
    # Combine
    Total_R = R_333 + R_334
    
    U_final = np.hstack([U_333, U_334])
    V_final = np.hstack([V_333, V_334])
    
    W_final = np.zeros((21, Total_R))
    # Copy W_333 to rows 0..8 (k=0,1,2)
    W_final[0:9, 0:R_333] = W_333
    # Copy W_334 to rows 9..20 (k=3,4,5,6)
    # W_334 already has indices (k+3)*3+i.
    # So W_334[row, col] should go to W_final[row, R_333+col]
    W_final[:, R_333:Total_R] = W_334
    
    # Convert to lists
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
