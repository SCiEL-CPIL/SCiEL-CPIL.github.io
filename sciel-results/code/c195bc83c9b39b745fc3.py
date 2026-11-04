import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Rank-42 Strassen-based decomposition: 6 blocks of 2x2x2, each with rank 7
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_A = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    strassen_B = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    w_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for b in range(6):
        s = b // 3
        k_block = b % 3
        
        for t in range(7):
            c = 7 * b + t
            
            u_coeff = strassen_A[t]
            for i in range(2):
                for j_loc in range(2):
                    idx_A = i * 2 + j_loc
                    coeff = u_coeff[idx_A]
                    if coeff != 0:
                        global_i = i
                        global_j = 2 * s + j_loc
                        U[global_i * m + global_j][c] = coeff
                        
            v_coeff = strassen_B[t]
            for j_loc in range(2):
                for k_loc in range(2):
                    idx_B = j_loc * 2 + k_loc
                    coeff = v_coeff[idx_B]
                    if coeff != 0:
                        global_j = 2 * s + j_loc
                        global_k = 2 * k_block + k_loc
                        V[global_j * p + global_k][c] = coeff
                        
            w_coeff = w_coeffs[t]
            for i in range(2):
                for k_loc in range(2):
                    idx_W = i * 2 + k_loc
                    coeff = w_coeff[idx_W]
                    if coeff != 0:
                        global_k = 2 * k_block + k_loc
                        global_i = i
                        W[global_k * n + global_i][c] = coeff
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        for i in range(n):
            for k in range(p):
                for a in range(n):
                    for b in range(m):
                        for c in range(m):
                            for d in range(p):
                                expected = 1.0 if (a == i and c == b and d == k) else 0.0
                                actual = np.sum(W_np[k * n + i, :] * U_np[a * m + b, :] * V_np[c * p + d, :])
                                if abs(actual - expected) > 1e-9:
                                    return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook rank 48
        rank = 48
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t = i * m * p + j * p + k
                    U[i * m + j][t] = 1
                    V[j * p + k][t] = 1
                    W[k * n + i][t] = 1
    
    return {"U": U, "V": V, "W": W}
