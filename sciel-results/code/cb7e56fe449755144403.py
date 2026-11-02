import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 2, 13
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    try:
        # Strassen-based decomposition: 6 blocks of 2x2 (rank 7 each) + 1 block of 2x1 (rank 4)
        rank_strassen = 6 * 7 + 4
        U = np.zeros((n * m, rank_strassen))
        V = np.zeros((m * p, rank_strassen))
        W = np.zeros((p * n, rank_strassen))
        
        # Correct Strassen terms for C = A * B
        # A indices: 0:a00, 1:a01, 2:a10, 3:a11
        # B indices: 0:b00, 1:b01, 2:b10, 3:b11
        # C indices: 0:c00, 1:c01, 2:c10, 3:c11
        
        strassen_terms = [
            # M1
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            # M2
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 1, 0]),
            # M3
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, -1]),
            # M4
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, -1, 0]),
            # M5
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            # M6
            ([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 1, 1]),
            # M7
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]),
        ]
        
        col = 0
        for block in range(6):
            k0 = block * 2
            k1 = block * 2 + 1
            
            for t in range(7):
                u_local, v_local, w_local = strassen_terms[t]
                
                # U: global index = i*m + j
                U[0, col] = u_local[0]
                U[1, col] = u_local[1]
                U[2, col] = u_local[2]
                U[3, col] = u_local[3]
                
                # V: global index = j*p + k
                V[0 * p + k0, col] = v_local[0]
                V[0 * p + k1, col] = v_local[1]
                V[1 * p + k0, col] = v_local[2]
                V[1 * p + k1, col] = v_local[3]
                
                # W: global index = k*n + i
                W[k0 * n + 0, col] = w_local[0]
                W[k0 * n + 1, col] = w_local[1]
                W[k1 * n + 0, col] = w_local[2]
                W[k1 * n + 1, col] = w_local[3]
                
                col += 1
        
        # Last column (k=12): schoolbook 2x2 x 2x1, rank 4
        k_last = 12
        for i in range(n):
            for j in range(m):
                U[i * m + j, col] = 1.0
                V[j * p + k_last, col] = 1.0
                W[k_last * n + i, col] = 1.0
                col += 1
        
        # Verify correctness
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i * m + j, j * p + k, k * n + i] = 1.0
                    
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
