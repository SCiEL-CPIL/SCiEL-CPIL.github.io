import numpy as np
import time

def construct():
    deadline = time.time() + 110
    n, m, p = 3, 4, 4
    
    def verify(U, V, W):
        rank = U.shape[1]
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        
        T_exp = np.zeros((n*m, m*p, p*n))
        for i in range(n*m):
            ar, ac = i // m, i % m
            for j in range(m*p):
                br, bc = j // p, j % p
                if ac == br:
                    k = bc * n + ar
                    T_exp[i, j, k] = 1
        
        return np.allclose(T, T_exp, atol=1e-9)
    
    # Construct rank 44 solution
    U44 = np.zeros((n * m, 44))
    V44 = np.zeros((m * p, 44))
    W44 = np.zeros((p * n, 44))
    
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen part (cols 0-6) for 2x2x2
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
    
    # Trivial part (cols 7-10) for the 3rd row of the 3x2x2 block
    U_loc[4, 7] = 1; V_loc[0, 7] = 1; W_loc[4, 7] = 1
    U_loc[5, 8] = 1; V_loc[2, 8] = 1; W_loc[4, 8] = 1
    U_loc[4, 9] = 1; V_loc[1, 9] = 1; W_loc[5, 9] = 1
    U_loc[5, 10] = 1; V_loc[3, 10] = 1; W_loc[5, 10] = 1
    
    def add_block(U, V, W, block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_idx = (a_row_start + r) * m + (a_col_start + c)
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_idx = (c_col_start + c) * n + (c_row_start + r)
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]
        
    # 4 blocks of 3x2x2
    add_block(U44, V44, W44, 0, 0, 0, 0, 0, 0, 0)
    add_block(U44, V44, W44, 1, 0, 2, 2, 0, 0, 0)
    add_block(U44, V44, W44, 2, 0, 0, 0, 2, 0, 2)
    add_block(U44, V44, W44, 3, 0, 2, 2, 2, 0, 2)
    
    if verify(U44, V44, W44):
        return {"U": U44.tolist(), "V": V44.tolist(), "W": W44.tolist()}
    
    # Fallback: Rank 48
    U_f = np.zeros((n*m, 48))
    V_f = np.zeros((m*p, 48))
    W_f = np.zeros((p*n, 48))
    r = 0
    for i in range(n*m):
        ar, ac = i // m, i % m
        for j in range(m*p):
            br, bc = j // p, j % p
            if ac == br:
                k = bc * n + ar
                U_f[i, r] = 1
                V_f[j, r] = 1
                W_f[k, r] = 1
                r += 1
    return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
