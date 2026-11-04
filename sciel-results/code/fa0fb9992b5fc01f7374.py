import numpy as np
from collections import defaultdict
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = 4, 4, 8
    
    # Build Strassen 2x2x2 base
    U_2 = np.zeros((4, 7), dtype=np.int8)
    V_2 = np.zeros((4, 7), dtype=np.int8)
    W_2 = np.zeros((4, 7), dtype=np.int8)
    
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    A_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (1, 3)],
        [(1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (-1, 0)],
        [(1, 1), (-1, 3)],
    ]
    
    B_specs = [
        [(1, 0), (1, 3)],
        [(1, 0)],
        [(1, 1), (-1, 3)],
        [(1, 2), (-1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (1, 3)],
    ]
    
    C_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (-1, 3)],
        [(1, 1), (1, 3)],
        [(1, 0), (1, 2)],
        [(-1, 0), (1, 1)],
        [(1, 3)],
        [(1, 0)],
    ]
    
    # Build the full 98-rank decomposition
    rank_initial = 98
    U = np.zeros((n * m, rank_initial), dtype=np.int8)
    V = np.zeros((m * p, rank_initial), dtype=np.int8)
    W = np.zeros((p * n, rank_initial), dtype=np.int8)
    
    col = 0
    for half in range(2):
        for k_p in range(7):
            for m_s in range(7):
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_a_idx = global_r * 4 + global_c
                            a_eff_idx = r * 2 + c
                            val = coeff_a * int(U_2[a_eff_idx, m_s])
                            if val != 0:
                                U[global_a_idx, col] += val
                
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_k = global_c + half * 4
                            global_b_idx = global_r * 8 + global_k
                            b_eff_idx = r * 2 + c
                            val = coeff_b * int(V_2[b_eff_idx, m_s])
                            if val != 0:
                                V[global_b_idx, col] += val
                
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_i = bi * 2 + r
                            global_k = bj * 2 + c + half * 4
                            global_w_idx = global_k * 4 + global_i
                            p_eff_idx = r * 2 + c
                            val = coeff_c * int(W_2[p_eff_idx, m_s])
                            if val != 0:
                                W[global_w_idx, col] += val
                
                col += 1
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, n, m, p):
        """Check if U ⊗ V ⊗ W decomposes the matrix multiplication tensor."""
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # Should equal 1 if j==k (for A[i,j]*B[j,k]), 0 otherwise
        U_arr = np.array(U_mat, dtype=np.float64)
        V_arr = np.array(V_mat, dtype=np.float64)
        W_arr = np.array(W_mat, dtype=np.float64)
        
        rank = U_arr.shape[1]
        # Tensor T has shape (n*m, m*p, p*n) with T[i, j, k] where:
        # i = row_A * m + col_A
        # j = row_B * p + col_B  
        # k = col_C * n + row_C
        # T[i,j,k] = 1 if col_A == row_B and col_B == row_C, else 0
        
        for i in range(n * m):
            row_A, col_A = divmod(i, m)
            for j in range(m * p):
                row_B, col_B = divmod(j, p)
                if col_A != row_B:
                    # All k should give 0
                    for k in range(p * n):
                        val = np.sum(U_arr[i, :] * V_arr[j, :] * W_arr[k, :])
                        if abs(val) > 1e-9:
                            return False
                else:
                    for k in range(p * n):
                        col_C, row_C = divmod(k, n)
                        expected = 1.0 if col_B == row_C else 0.0
                        val = np.sum(U_arr[i, :] * V_arr[j, :] * W_arr[k, :])
                        if abs(val - expected) > 1e-9:
                            return False
        return True
    
    if not verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        # Fallback: return schoolbook rank 128
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school), dtype=np.int8)
        V_s = np.zeros((m * p, rank_school), dtype=np.int8)
        W_s = np.zeros((p * n, rank_school), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, idx] = 1
                    V_s[j * p + k, idx] = 1
                    W_s[k * n + i, idx] = 1
                    idx += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Try to reduce rank by finding linear dependencies
    # The tensor is T = sum_r U_r ⊗ V_r ⊗ W_r
    # If there's a linear relation sum_r alpha_r * (U_r ⊗ V_r ⊗ W_r) = 0,
    # we can remove one term.
    
    # Flatten each rank-1 tensor into a vector and find null space
    # This is expensive for 98 terms in high-dimensional space, but let's try
    
    # Alternative: check if any column of U, V, or W is linearly dependent
    # with others in a way that allows rank reduction
    
    # For now, return the rank-98 decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
