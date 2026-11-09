import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def get_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices:
    U: i*m+j -> i*2+j
    V: j*p+k -> j*2+k
    W: k*n+i -> k*2+i
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Term 0: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1 # A00, A11 (indices 0, 3 in 2x2 flat)
    V[0][0] = 1; V[3][0] = 1 # B00, B11 (indices 0, 3 in 2x2 flat)
    W[0][0] = 1; W[3][0] = 1 # C00, C11 (indices 0, 3 in 2x2 flat: C00=0, C01=1, C10=2, C11=3? 
                               # Wait, W index is k*n+i.
                               # C00: k=0, i=0 -> 0
                               # C01: k=1, i=0 -> 1
                               # C10: k=0, i=1 -> 2
                               # C11: k=1, i=1 -> 3
                               # So W[0] is C00, W[1] is C01, W[2] is C10, W[3] is C11.
    W[0][0] = 1
    W[3][0] = 1
    
    # Term 1: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1 # A10 (idx 2), A11 (idx 3)
    V[0][1] = 1
    W[2][1] = 1 # C10
    W[3][1] = -1 # C11
    
    # Term 2: A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1 # B10 (idx 2), B11 (idx 3)
    W[1][2] = 1 # C01
    W[3][2] = 1 # C11
    
    # Term 3: A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1 # B01 (idx 1), B00 (idx 0)
    W[0][3] = 1 # C00
    W[2][3] = 1 # C10
    
    # Term 4: (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1 # A00 (0), A01 (1)
    V[3][4] = 1
    W[0][4] = -1 # C00
    W[1][4] = 1 # C01
    
    # Term 5: (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1 # A10 (2), A00 (0)
    V[0][5] = 1; V[1][5] = 1 # B00 (0), B01 (1)
    W[3][5] = 1 # C11
    
    # Term 6: (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1 # A01 (1), A11 (3)
    V[2][6] = 1; V[3][6] = 1 # B10 (2), B11 (3)
    W[0][6] = 1 # C00
    W[1][6] = 1 # C01
    
    return U, V, W

def get_schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Partitions
    I_parts = [[0, 1], [2, 3]]
    J_parts = [[0, 1], [2, 3], [4]]
    K_parts = [[0, 1], [2, 3], [4]]
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    for i_idx, I in enumerate(I_parts):
        n_i = len(I)
        for j_idx, J in enumerate(J_parts):
            m_j = len(J)
            for k_idx, K in enumerate(K_parts):
                p_k = len(K)
                
                if n_i == 2 and m_j == 2 and p_k == 2:
                    U_sub, V_sub, W_sub = get_strassen_2x2x2()
                else:
                    U_sub, V_sub, W_sub = get_schoolbook(n_i, m_j, p_k)
                
                # Map local indices to global
                # U local: i_local * m_j + j_local
                # U global: I[i_local] * m + J[j_local]
                U_global = [[0]*len(U_sub[0]) for _ in range(n*m)]
                for i_l in range(n_i):
                    for j_l in range(m_j):
                        global_i = I[i_l]
                        global_j = J[j_l]
                        global_idx = global_i * m + global_j
                        local_idx = i_l * m_j + j_l
                        for c in range(len(U_sub[0])):
                            U_global[global_idx][c] = U_sub[local_idx][c]
                
                V_global = [[0]*len(V_sub[0]) for _ in range(m*p)]
                for j_l in range(m_j):
                    for k_l in range(p_k):
                        global_j = J[j_l]
                        global_k = K[k_l]
                        global_idx = global_j * p + global_k
                        local_idx = j_l * p_k + k_l
                        for c in range(len(V_sub[0])):
                            V_global[global_idx][c] = V_sub[local_idx][c]
                
                W_global = [[0]*len(W_sub[0]) for _ in range(p*n)]
                for k_l in range(p_k):
                    for i_l in range(n_i):
                        global_k = K[k_l]
                        global_i = I[i_l]
                        global_idx = global_k * n + global_i
                        local_idx = k_l * n_i + i_l
                        for c in range(len(W_sub[0])):
                            W_global[global_idx][c] = W_sub[local_idx][c]
                            
                U_blocks.append(U_global)
                V_blocks.append(V_global)
                W_blocks.append(W_global)
                
    # Concatenate
    total_rank = sum(len(b[0]) for b in U_blocks)
    U_final = [[0]*total_rank for _ in range(n*m)]
    V_final = [[0]*total_rank for _ in range(m*p)]
    W_final = [[0]*total_rank for _ in range(p*n)]
    
    offset = 0
    for i_b in range(len(U_blocks)):
        U_b = U_blocks[i_b]
        V_b = V_blocks[i_b]
        W_b = W_blocks[i_b]
        r_b = len(U_b[0])
        for i in range(n*m):
            for c in range(r_b):
                U_final[i][offset + c] = U_b[i][c]
        for i in range(m*p):
            for c in range(r_b):
                V_final[i][offset + c] = V_b[i][c]
        for i in range(p*n):
            for c in range(r_b):
                W_final[i][offset + c] = W_b[i][c]
        offset += r_b
        
    return {"U": U_final, "V": V_final, "W": W_final}
