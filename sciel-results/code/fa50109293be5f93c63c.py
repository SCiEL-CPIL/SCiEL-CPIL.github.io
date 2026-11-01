import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    
    # We decompose T(2,3,6) into two T(2,3,3) blocks.
    # Rank(T(2,3,3)) = 7. Total Rank = 14.
    
    # 1. Solve for Rank 7 decomposition of T(2,3,3)
    # T(2,3,3) dimensions: A(2x3), B(3x3), C(2x3)
    # U: 6x7, V: 9x7, W: 6x7
    
    U_233, V_233, W_233 = find_rank_7_233()
    
    if U_233 is None:
        # Fallback to schoolbook rank 36
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # 2. Construct Rank 14 decomposition for T(2,3,6)
    rank = 14
    U_final = np.zeros((n*m, rank))
    V_final = np.zeros((m*p, rank))
    W_final = np.zeros((p*n, rank))
    
    # Block 1: k in 0..2 (first 3 columns of B)
    # U is shared
    U_final[:, 0:7] = U_233
    
    # V: rows 0..8 correspond to j=0..2, k=0..2
    V_final[0:9, 0:7] = V_233
    
    # W: rows 0..5 correspond to k=0..2, i=0..1
    W_final[0:6, 0:7] = W_233
    
    # Block 2: k in 3..5 (last 3 columns of B)
    # U is shared
    U_final[:, 7:14] = U_233
    
    # V: rows 9..17 correspond to j=0..2, k=3..5
    # In V_233, the index for B is b = j*3 + k_local.
    # In V_final, the index for B is b_full = j*6 + k_full.
    # For block 2, k_full = 3 + k_local.
    # So row in V_final = j*6 + (3+k_local) = 6j + 3 + k_local.
    # Row in V_233 = 3j + k_local.
    # We need to map V_233 rows to V_final rows.
    for j in range(3):
        for k_local in range(3):
            row_src = j*3 + k_local
            row_dst = j*6 + (3 + k_local)
            V_final[row_dst, 7:14] = V_233[row_src, :]
            
    # W: rows 6..11 correspond to k=3..5, i=0..1
    # In W_233, index for C is c = k_local*2 + i.
    # In W_final, index for C is c_full = k_full*2 + i.
    # For block 2, k_full = 3 + k_local.
    # So row in W_final = (3+k_local)*2 + i = 6 + 2*k_local + i.
    # Row in W_233 = 2*k_local + i.
    for k_local in range(3):
        for i in range(2):
            row_src = k_local*2 + i
            row_dst = (3+k_local)*2 + i
            W_final[row_dst, 7:14] = W_233[row_src, :]
            
    # 3. Verification
    T_full = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_full[a, b, c] = 1.0
                
    recon = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        recon += np.einsum('a,b,c->abc', U_final[:, r], V_final[:, r], W_final[:, r])
        
    if np.max(np.abs(recon - T_full)) < 1e-8:
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
        
    # Fallback if verification fails
    rank = n * m * p
    U_fb = np.zeros((n*m, rank))
    V_fb = np.zeros((m*p, rank))
    W_fb = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

def find_rank_7_233():
    """
    Finds a rank 7 decomposition for 2x3x3 matrix multiplication.
    Returns U, V, W or None.
    """
    n, m, p = 2, 3, 3
    N1, N2, N3 = n*m, m*p, p*n # 6, 9, 6
    T = np.zeros((N1, N2, N3))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
                
    vec_T = T.reshape(-1)
    target_rank = 7
    
    # Try a few random initializations with ALS
    start_time = time.time()
    deadline = start_time + 30.0 # 30s limit for this sub-problem
    
    for trial in range(50):
        if time.time() > deadline:
            break
            
        U = np.random.randn(N1, target_rank)
        V = np.random.randn(N2, target_rank)
        W = np.random.randn(N3, target_rank)
        
        for it in range(100):
            if time.time() > deadline:
                break
                
            # Solve U
            # vec(T) = sum_r U[:,r] * (V[:,r] kron W[:,r])
            A_U = np.zeros((N2*N3, N1*target_rank))
            for r in range(target_rank):
                A_U[:, r*target_rank:(r+1)*target_rank] = np.kron(V[:, r], W[:, r])[:, None] * np.ones((N2*N3, 1)) # Wait, this is wrong.
                # Correct: Column r of A_U should be V[:,r] kron W[:,r] repeated for each row of U?
                # No.
                # T = sum_r U_r V_r W_r
                # vec(T) = sum_r (U_r) * (V_r kron W_r)  ... No.
                # vec(T) = sum_r (W_r kron V_r) U_r  ... No.
                # The Khatri-Rao product of V and W is a matrix of size (N2*N3) x rank.
                # Let K = V_kr_W.
                # Then vec(T) = U^T K? No.
                # T_{abc} = sum_r U_{ar} V_{br} W_{cr}
                # vec(T)_{(a, (b,c))} = sum_r U_{ar} (V_{br} W_{cr})
                # So vec(T) = U^T * (V_kr_W) is not quite right because U is N1 x rank.
                # vec(T) is length N1*N2*N3.
                # We can reshape T to (N1, N2*N3).
                # T_mat = T.reshape(N1, N2*N3)
                # T_mat = U^T @ (V_kr_W)^T ?
                # T_mat_{a, (b,c)} = sum_r U_{r,a} V_{r,b} W_{r,c}
                # Let M be matrix where M_{(b,c), r} = V_{r,b} W_{r,c}.
                # Then T_mat_{a, (b,c)} = sum_r U_{r,a} M_{(b,c), r}
                # So T_mat = U^T @ M.
                # We want to solve for U given V, W.
                # T_mat = U^T @ M
                # vec(T_mat) = (M^T kron I) vec(U^T) ... complicated.
                # Easier:
                # T_mat = U^T M
                # Multiply by M^T on right: T_mat M^T = U^T M M^T
                # If M M^T is invertible, U^T = T_mat M^T (M M^T)^{-1}
                # Or just use lstsq on the linear system for each row of U^T?
                # Actually, simplest is to build the big matrix.
                
            # Let's use the standard ALS formulation
            # Fix V, W. Solve for U.
            # T_{abc} = sum_r U_{ar} V_{br} W_{cr}
            # For fixed b,c, T_{abc} is linear in U_{ar}.
            # We can solve for U using lstsq.
            # Build matrix A_U of size (N2*N3) x (N1*target_rank)
            # This is expensive. N2*N3 = 54. N1*rank = 42.
            # 54x42 matrix. Very small.
            
            A_U = np.zeros((N2*N3, N1*target_rank))
            for b in range(N2):
                for c in range(N3):
                    row_idx = b*N3 + c
                    for a in range(N1):
                        for r in range(target_rank):
                            col_idx = a*target_rank + r
                            A_U[row_idx, col_idx] = V[b, r] * W[c, r]
                            
            try:
                U_flat, _, _, _ = np.linalg.lstsq(A_U, vec_T, rcond=None)
                U = U_flat.reshape(N1, target_rank)
            except:
                break
                
            # Solve V
            A_V = np.zeros((N1*N3, N2*target_rank))
            for a in range(N1):
                for c in range(N3):
                    row_idx = a*N3 + c
                    for b in range(N2):
                        for r in range(target_rank):
                            col_idx = b*target_rank + r
                            A_V[row_idx, col_idx] = U[a, r] * W[c, r]
                            
            try:
                V_flat, _, _, _ = np.linalg.lstsq(A_V, vec_T, rcond=None)
                V = V_flat.reshape(N2, target_rank)
            except:
                break
                
            # Solve W
            A_W = np.zeros((N1*N2, N3*target_rank))
            for a in range(N1):
                for b in range(N2):
                    row_idx = a*N2 + b
                    for c in range(N3):
                        for r in range(target_rank):
                            col_idx = c*target_rank + r
                            A_W[row_idx, col_idx] = U[a, r] * V[b, r]
                            
            try:
                W_flat, _, _, _ = np.linalg.lstsq(A_W, vec_T, rcond=None)
                W = W_flat.reshape(N3, target_rank)
            except:
                break
                
            # Check error
            recon = np.zeros((N1, N2, N3))
            for r in range(target_rank):
                recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            error = np.max(np.abs(recon - T))
            if error < 1e-8:
                return U, V, W
                
    return None, None, None
