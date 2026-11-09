import time
import sys
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # 1. Schoolbook (Rank 72)
    result_72 = _schoolbook_decomposition(n, m, p)
    
    # 2. Strassen Hybrid (Rank 66)
    result_66 = _strassen_hybrid_decomposition(n, m, p)
    
    # Validate
    if _is_valid(result_66, n, m, p):
        return result_66
        
    return result_72

def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_hybrid_decomposition(n, m, p):
    # n=3, m=4, p=6
    U = [[0] * 66 for _ in range(12)]
    V = [[0] * 66 for _ in range(24)]
    W = [[0] * 66 for _ in range(18)]
    
    # Strassen factors for 2x2x2
    U_s = [
        [1, 0, 0, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_s = [
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # 1x2x2 Schoolbook factors
    U_122 = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    V_122 = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    W_122 = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]

    col = 0
    
    def add_block_mapped(U_local, V_local, W_local, u_idx, v_idx, w_idx, rank_block):
        nonlocal col
        for r in range(rank_block):
            for l_i, g_i in enumerate(u_idx):
                U[g_i][col] += U_local[l_i][r]
            for l_j, g_j in enumerate(v_idx):
                V[g_j][col] += V_local[l_j][r]
            for l_k, g_k in enumerate(w_idx):
                W[g_k][col] += W_local[l_k][r]
            col += 1

    # Global index mappings
    u_A11 = [0, 1, 4, 5]
    u_A12 = [2, 3, 6, 7]
    u_A21 = [8, 9]
    u_A22 = [10, 11]
    
    v_B11 = [0, 1, 6, 7]
    v_B12 = [2, 3, 8, 9]
    v_B13 = [4, 5, 10, 11]
    v_B21 = [12, 13, 18, 19]
    v_B22 = [14, 15, 20, 21]
    v_B23 = [16, 17, 22, 23]
    
    w_C1 = [0, 1, 3, 4]
    w_C2 = [6, 7, 9, 10]
    w_C3 = [12, 13, 15, 16]
    
    w_D1 = [2, 5]
    w_D2 = [8, 11]
    w_D3 = [14, 17]
    
    # Top Part
    add_block_mapped(U_s, V_s, W_s, u_A11, v_B11, w_C1, 7)
    add_block_mapped(U_s, V_s, W_s, u_A12, v_B21, w_C1, 7)
    
    add_block_mapped(U_s, V_s, W_s, u_A11, v_B12, w_C2, 7)
    add_block_mapped(U_s, V_s, W_s, u_A12, v_B22, w_C2, 7)
    
    add_block_mapped(U_s, V_s, W_s, u_A11, v_B13, w_C3, 7)
    add_block_mapped(U_s, V_s, W_s, u_A12, v_B23, w_C3, 7)
    
    # Bottom Part
    add_block_mapped(U_122, V_122, W_122, u_A21, v_B11, w_D1, 4)
    add_block_mapped(U_122, V_122, W_122, u_A22, v_B21, w_D1, 4)
    
    add_block_mapped(U_122, V_122, W_122, u_A21, v_B12, w_D2, 4)
    add_block_mapped(U_122, V_122, W_122, u_A22, v_B22, w_D2, 4)
    
    add_block_mapped(U_122, V_122, W_122, u_A21, v_B13, w_D3, 4)
    add_block_mapped(U_122, V_122, W_122, u_A22, v_B23, w_D3, 4)
    
    return {"U": U, "V": V, "W": W}

def _is_valid(result, n, m, p):
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    rank = U.shape[1]
    
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
        
    T_recon = np.zeros((n*m, m*p, p*n))
    for c in range(rank):
        T_recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1
                
    return np.allclose(T_recon, T_target)

# EVOLVE-BLOCK-END
