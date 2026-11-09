import time
import numpy as np
import random

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 140

    # Generate 4x4x4 Strassen decomposition
    def get_block_indices(r, c, size, total_size):
        idxs = []
        for i in range(r*size, (r+1)*size):
            for j in range(c*size, (c+1)*size):
                idxs.append(i*total_size + j)
        return idxs

    def strassen_decomp(n):
        if n == 1:
            return np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]])
        
        m_sub = n // 2
        U_m, V_m, W_m = strassen_decomp(m_sub)
        rank_m = U_m.shape[1]
        
        U_n = np.zeros((n*n, 7 * rank_m))
        V_n = np.zeros((n*n, 7 * rank_m))
        W_n = np.zeros((n*n, 7 * rank_m))
        
        A_coeffs = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]]
        ]
        B_coeffs = [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [1, -1]],
            [[-1, 1], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]]
        ]
        
        C_coeffs = {
            (0,0): [1, 0, 0, 1, -1, 0, 1],
            (0,1): [0, 0, 1, 0, 1, 0, 0],
            (1,0): [0, 1, 0, 1, 0, 0, 0],
            (1,1): [1, -1, 1, 0, 0, 1, 0]
        }
        
        for t in range(7):
            for k in range(rank_m):
                col_idx = t * rank_m + k
                
                # Fill U_n
                for i in range(m_sub):
                    for j in range(m_sub):
                        u_weight = U_m[i*m_sub + j, k]
                        if u_weight != 0:
                            for r in range(2):
                                for c in range(2):
                                    a_coeff = A_coeffs[t][r][c]
                                    if a_coeff != 0:
                                        idx = (r*m_sub + i) * n + (c*m_sub + j)
                                        U_n[idx, col_idx] += u_weight * a_coeff
                                        
                # Fill V_n
                for i in range(m_sub):
                    for j in range(m_sub):
                        v_weight = V_m[i*m_sub + j, k]
                        if v_weight != 0:
                            for r in range(2):
                                for c in range(2):
                                    b_coeff = B_coeffs[t][r][c]
                                    if b_coeff != 0:
                                        idx = (r*m_sub + i) * n + (c*m_sub + j)
                                        V_n[idx, col_idx] += v_weight * b_coeff
                                        
                # Fill W_n
                for i in range(m_sub):
                    for j in range(m_sub):
                        w_weight = W_m[i*m_sub + j, k]
                        if w_weight != 0:
                            for (rc_r, rc_c), coeffs in C_coeffs.items():
                                c_coeff = coeffs[t]
                                if c_coeff != 0:
                                    idx = (rc_c*m_sub + j) * n + (rc_r*m_sub + i)
                                    W_n[idx, col_idx] += w_weight * c_coeff

        return U_n, V_n, W_n

    # Get 4x4x4 decomposition
    U_4, V_4, W_4 = strassen_decomp(4)
    rank_4 = U_4.shape[1] # Should be 49
    
    # Construct 4x4x8 decomposition
    rank_full = 2 * rank_4
    U_full = np.zeros((n*m, rank_full))
    V_full = np.zeros((m*p, rank_full))
    W_full = np.zeros((p*n, rank_full))
    
    # First half: B_L
    U_full[:, 0:rank_4] = U_4
    V_full[0:m*4, 0:rank_4] = V_4
    # W index for C_L (k in 0..3) is k*4+i
    W_full[0:4*n, 0:rank_4] = W_4
    
    # Second half: B_R
    U_full[:, rank_4:rank_full] = U_4
    # V index for B_R (k in 4..7) is j*p+k. j in 0..3, k in 4..7.
    # V_4 indices are j*4+k' where k' in 0..3.
    # So V_4 should be placed in rows 16..31 of V_full.
    V_full[m*4:m*p, rank_4:rank_full] = V_4
    # W index for C_R (k in 4..7) is k*4+i.
    # W_4 indices are k'*4+i where k' in 0..3.
    # So W_4 should be placed in rows 16..31 of W_full.
    W_full[4*n:8*n, rank_4:rank_full] = W_4
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    # Verify
    def verify(U_list, V_list, W_list, tol=1e-9):
        U = np.array(U_list, dtype=float)
        V = np.array(V_list, dtype=float)
        W = np.array(W_list, dtype=float)
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    if verify(U_list, V_list, W_list):
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        rank_school = n * m * p
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
