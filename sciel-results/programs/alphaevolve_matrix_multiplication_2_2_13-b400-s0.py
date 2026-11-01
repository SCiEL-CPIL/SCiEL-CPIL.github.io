"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 2, 13
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_school[i * m + k, col] = 1.0
                V_school[k * p + j, col] = 1.0
                W_school[j * n + i, col] = 1.0
                col += 1
                
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    try:
        # Strassen-based decomposition: 6 blocks of 2x2 (rank 7 each) + 1 block of 2x1 (rank 4)
        rank_strassen = 6 * 7 + 4
        U = np.zeros((n * m, rank_strassen))
        V = np.zeros((m * p, rank_strassen))
        W = np.zeros((p * n, rank_strassen))
        
        # Strassen terms for 2x2x2x2: (u_t, v_t, w_t)
        # A indices: [a00, a01, a10, a11]
        # B indices: [b00, b01, b10, b11]
        # C indices: [c00, c10, c01, c11] (W index = j*n+i)
        strassen_terms = [
            (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
            (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1])),
            (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1])),
            (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0])),
            (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0])),
            (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
            (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0])),
        ]
        
        col = 0
        for block in range(6):
            j0 = block * 2
            j1 = block * 2 + 1
            
            for t in range(7):
                u_local, v_local, w_local = strassen_terms[t]
                
                # A: global index = i*m + k, local = [a00,a01,a10,a11]
                U[0, col] = u_local[0]
                U[1, col] = u_local[1]
                U[2, col] = u_local[2]
                U[3, col] = u_local[3]
                
                # B: global index = k*p + j, local = [b00,b01,b10,b11]
                V[0 * p + j0, col] = v_local[0]
                V[0 * p + j1, col] = v_local[1]
                V[1 * p + j0, col] = v_local[2]
                V[1 * p + j1, col] = v_local[3]
                
                # C: global index = j*n + i, local = [c00,c10,c01,c11]
                W[j0 * n + 0, col] = w_local[0]
                W[j0 * n + 1, col] = w_local[1]
                W[j1 * n + 0, col] = w_local[2]
                W[j1 * n + 1, col] = w_local[3]
                
                col += 1
        
        # Last column (j=12): schoolbook 2x2 x 2x1, rank 4
        j_last = 12
        for i in range(n):
            for k in range(m):
                U[i * m + k, col] = 1.0
                V[k * p + j_last, col] = 1.0
                W[j_last * n + i, col] = 1.0
                col += 1
        
        # Verify correctness
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T_target[i * m + k, k * p + j, j * n + i] = 1.0
                    
        T_approx = np.einsum('it,jt,kt->ijk', U, V, W)
        error = np.max(np.abs(T_approx - T_target))
        
        if error < 1e-10:
            best_rank = rank_strassen
            best_U, best_V, best_W = U, V, W
    
    except Exception:
        pass
    
    if time.time() > deadline:
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
