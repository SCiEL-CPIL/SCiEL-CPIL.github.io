import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 3, 8
    
    # Fallback Rank 44 Construction
    def get_rank_44():
        U = np.zeros((n*m, 44), dtype=np.int64)
        V = np.zeros((m*p, 44), dtype=np.int64)
        W = np.zeros((p*n, 44), dtype=np.int64)
        
        col_offset = 0
        for block in range(4):
            k0 = block * 2
            k1 = block * 2 + 1
            
            # Block 1: A[:, 0] * B[0, k0:k1]
            U[0*m + 0, col_offset] = 1
            V[0*p + k0, col_offset] = 1
            W[k0*n + 0, col_offset] = 1
            col_offset += 1
            
            U[0*m + 0, col_offset] = 1
            V[0*p + k1, col_offset] = 1
            W[k1*n + 0, col_offset] = 1
            col_offset += 1
            
            U[1*m + 0, col_offset] = 1
            V[0*p + k0, col_offset] = 1
            W[k0*n + 1, col_offset] = 1
            col_offset += 1
            
            U[1*m + 0, col_offset] = 1
            V[0*p + k1, col_offset] = 1
            W[k1*n + 1, col_offset] = 1
            col_offset += 1
            
            # Block 2: A[:, 1:3] * B[1:3, k0:k1] (Strassen-like for 2x2 part)
            # M1
            U[0*m + 1, col_offset] = 1
            U[1*m + 2, col_offset] = 1
            V[1*p + k0, col_offset] = 1
            V[2*p + k1, col_offset] = 1
            W[k0*n + 0, col_offset] = 1
            W[k1*n + 1, col_offset] = 1
            col_offset += 1
            
            # M2
            U[1*m + 1, col_offset] = 1
            U[1*m + 2, col_offset] = 1
            V[1*p + k0, col_offset] = 1
            W[k0*n + 1, col_offset] = 1
            W[k1*n + 1, col_offset] = -1
            col_offset += 1
            
            # M3
            U[0*m + 1, col_offset] = 1
            V[1*p + k1, col_offset] = 1
            V[2*p + k1, col_offset] = -1
            W[k1*n + 0, col_offset] = 1
            W[k1*n + 1, col_offset] = 1
            col_offset += 1
            
            # M4
            U[1*m + 2, col_offset] = 1
            V[2*p + k0, col_offset] = 1
            V[1*p + k0, col_offset] = -1
            W[k0*n + 0, col_offset] = 1
            W[k0*n + 1, col_offset] = 1
            col_offset += 1
            
            # M5
            U[0*m + 1, col_offset] = 1
            U[0*m + 2, col_offset] = 1
            V[2*p + k1, col_offset] = 1
            W[k0*n + 0, col_offset] = -1
            W[k1*n + 0, col_offset] = 1
            col_offset += 1
            
            # M6
            U[1*m + 1, col_offset] = 1
            U[0*m + 1, col_offset] = -1
            V[1*p + k0, col_offset] = 1
            V[1*p + k1, col_offset] = 1
            W[k1*n + 1, col_offset] = 1
            col_offset += 1
            
            # M7
            U[0*m + 2, col_offset] = 1
            U[1*m + 2, col_offset] = -1
            V[2*p + k0, col_offset] = 1
            V[2*p + k1, col_offset] = 1
            W[k0*n + 0, col_offset] = 1
            col_offset += 1
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Target: Find rank for 2x3x2
    # If found, total rank is 4 * R_sub
    # Try R_sub = 7, 8, 9
    
    def find_sub_rank(R_sub):
        # T_sub for 2x3x2
        # U: 2xR, V: 3xR, W: 2xR
        # T[i, j, k] = 1 if i in {0,1}, j in {0,1,2}, k in {0,1}
        # C[i, k] = sum_j A[i, j] B[j, k]
        
        # We search for integer solutions in [-2, 2]
        # This is a small search space if we use constraints.
        # But 7 variables per vector, 3 vectors, 7 columns.
        # Too large for brute force.
        
        # Use ALS with integer rounding
        best_score = np.inf
        best_solution = None
        
        for _ in range(50): # 50 restarts
            if time.time() > deadline - 20:
                break
                
            U = np.random.randint(-2, 3, (2, R_sub)).astype(float)
            V = np.random.randint(-2, 3, (3, R_sub)).astype(float)
            W = np.random.randint(-2, 3, (2, R_sub)).astype(float)
            
            # ALS iterations
            for it in range(20):
                # Update U
                # T_unfolded_1 (2, 3*2) = U @ (V outer W)
                # V_W = np.einsum('bd,cd->bc', V, W) # 3x2
                # T_1 = T.reshape(2, 6)
                # B_mat (6, R)
                B_mat = np.einsum('bd,cd->bc', V, W).reshape(3*2, R_sub)
                T_1 = np.zeros((2, 6))
                T_1[0, 0] = 1; T_1[0, 1] = 1; T_1[0, 2] = 1
                T_1[1, 3] = 1; T_1[1, 4] = 1; T_1[1, 5] = 1
                # T[i, j*2+k] = delta_ij? No.
                # T[i, j, k] = 1 for all i,j,k? No.
                # The tensor is the multiplication tensor.
                # T[i, j, k] is the coefficient of A[i,j] B[j,k] in C[i,k].
                # It is 1 for all i,j,k.
                # So T is all 1s?
                # No, T is a tensor of size 2x3x2.
                # T[i, j, k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
                # Yes, it contributes. So T[i,j,k] = 1 for all i,j,k.
                
                T_1 = np.ones((2, 6))
                
                try:
                    U_new, _, _, _ = np.linalg.lstsq(B_mat.T, T_1.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                
                # Update V
                A_mat = np.einsum('ad,cd->ac', U, W).reshape(2*2, R_sub)
                T_2 = np.ones((3, 4)) # T.transpose(1,0,2) -> (3, 2, 2) -> (3, 4)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(A_mat.T, T_2.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                
                # Update W
                C_mat = np.einsum('ad,bd->ab', U, V).reshape(2*3, R_sub)
                T_3 = np.ones((2, 6)) # T.transpose(2,0,1) -> (2, 2, 3) -> (2, 6)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(C_mat.T, T_3.T, rcond=None)
                    W = W_new.T
                except:
                    pass
                    
            # Round and verify
            U_r = np.round(U)
            V_r = np.round(V)
            W_r = np.round(W)
            
            # Verify
            T_check = np.einsum('ad,bd,cd->abc', U_r, V_r, W_r)
            T_exp = np.ones((2, 3, 2))
            if np.allclose(T_check, T_exp, atol=1e-9):
                return U_r, V_r, W_r
                
        return None

    # Try to find rank 7 for 2x3x2
    if time.time() < deadline - 30:
        sol = find_sub_rank(7)
        if sol:
            U_sub, V_sub, W_sub = sol
            # Construct full 2x3x8
            U_full = np.zeros((2*3, 4*7), dtype=np.int64)
            V_full = np.zeros((3*8, 4*7), dtype=np.int64)
            W_full = np.zeros((8*2, 4*7), dtype=np.int64)
            
            for l in range(4):
                k0 = l * 2
                # U is shared?
                # In the decomposition T = sum U_d V_d W_d, U_d is the factor for A.
                # If we use the same U_d for all blocks, we just add the terms.
                # Term d for block l: U_d (from sub) * V_{l,d} * W_{l,d}
                # U_full[:, l*7 + d] = U_sub[:, d]
                # V_full[:, l*7 + d] = 0 except for rows corresponding to block l
                # W_full[:, l*7 + d] = 0 except for rows corresponding to block l
                
                for d in range(7):
                    idx = l * 7 + d
                    U_full[:, idx] = U_sub[:, d]
                    
                    # V_sub is 3x7. Rows correspond to j in {0,1,2}.
                    # In full V, rows are j*p + k.
                    # For block l, k is k0, k0+1.
                    # V_full[j*p + k, idx] = V_sub[j, d]
                    for j in range(3):
                        for k in range(2):
                            V_full[j*8 + k0 + k, idx] = V_sub[j, d]
                            
                    # W_sub is 2x7. Rows correspond to i in {0,1}.
                    # In full W, rows are k*n + i.
                    # For block l, k is k0, k0+1.
                    # W_full[k*n + i, idx] = W_sub[i, d]
                    for k in range(2):
                        for i in range(2):
                            W_full[(k0 + k)*2 + i, idx] = W_sub[i, d]
                            
            # Verify full solution
            T_check = np.einsum('ad,bd,cd->abc', U_full, V_full, W_full)
            T_exp = np.zeros((6, 24, 16))
            for i in range(2):
                for j in range(3):
                    for k in range(8):
                        a_idx = i * 3 + j
                        b_idx = j * 8 + k
                        c_idx = k * 2 + i
                        T_exp[a_idx, b_idx, c_idx] = 1
                        
            if np.allclose(T_check, T_exp, atol=1e-9):
                return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback
    return get_rank_44()
