import numpy as np
import time

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Local factors for 3x2x2 multiplication
    # A is 3x2, B is 2x2, C is 3x2
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen part (cols 0-6) for the 2x2x2 core (rows 0,1 of A)
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    # Extra part (cols 7-10) for Row 2 of A
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1
    
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1
    
    # Mapping function to place local blocks into global matrices
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11
        
        for r in range(3): 
            for c in range(2): 
                local_a_idx = r * 2 + c
                global_a_row = a_row_start + r
                global_a_col = a_col_start + c
                global_a_idx = global_a_row * m + global_a_col
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
                
        for r in range(2): 
            for c in range(2): 
                local_b_idx = r * 2 + c
                global_b_row = b_row_start + r
                global_b_col = b_col_start + c
                global_b_idx = global_b_row * p + global_b_col
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
                
        for r in range(3): 
            for c in range(2): 
                local_c_idx = r * 2 + c
                global_c_row = c_row_start + r
                global_c_col = c_col_start + c
                global_c_idx = global_c_col * n + global_c_row
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]

    # Block 0: A_L (cols 0,1) * B_11 (rows 0,1, cols 0,1) -> C_L (cols 0,1)
    add_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 1: A_R (cols 2,3) * B_21 (rows 2,3, cols 0,1) -> C_L (cols 0,1)
    add_block(1, 0, 2, 2, 0, 0, 0)
    
    # Block 2: A_L (cols 0,1) * B_12 (rows 0,1, cols 2,3) -> C_R (cols 2,3)
    add_block(2, 0, 0, 0, 2, 0, 2)
    
    # Block 3: A_R (cols 2,3) * B_22 (rows 2,3, cols 2,3) -> C_R (cols 2,3)
    add_block(3, 0, 2, 2, 2, 0, 2)
    
    # Verify correctness
    def verify(U_mat, V_mat, W_mat):
        Nm, Mp, Pn = n*m, m*p, p*n
        R = U_mat.shape[1]
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Expected: T[i*m+j, j*p+k, k*n+i] = 1 if valid, else 0
        UV = U_mat[:, None, :] * V_mat[None, :, :]  # (Nm, Mp, R)
        UV_flat = UV.reshape(-1, R)
        T_rec_flat = UV_flat @ W_mat.T  # (Nm*Mp, Pn)
        T_rec = T_rec_flat.reshape(Nm, Mp, Pn)
        
        # Build expected tensor
        T_exp = np.zeros((Nm, Mp, Pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_exp[i*m+j, j*p+k, k*n+i] = 1.0
        
        return np.allclose(T_rec, T_exp, atol=1e-8)
    
    if not verify(U, V, W):
        # Fallback to direct decomposition
        R = n * m * p
        U_dir = np.zeros((n*m, R))
        V_dir = np.zeros((m*p, R))
        W_dir = np.zeros((p*n, R))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_dir[i*m+j, idx] = 1.0
                    V_dir[j*p+k, idx] = 1.0
                    W_dir[k*n+i, idx] = 1.0
                    idx += 1
        return {"U": U_dir.tolist(), "V": V_dir.tolist(), "W": W_dir.tolist()}
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
