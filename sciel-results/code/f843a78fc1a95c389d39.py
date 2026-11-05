import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    # Strategy: Kronecker product of 2x2 Strassen (rank 7) and 3x3 schoolbook (rank 27).
    # Total rank = 7 * 27 = 189.
    # This decomposes 6x6 matmul by viewing it as 2x2 blocks of 3x3 submatrices.
    
    n, m, p = DIMENSIONS  # all 6
    rank = 7 * 27
    
    # 2x2 Strassen decomposition
    # U, V, W have shape (4, 7)
    # Indexing: row index = i*2 + j for A[i,j], B[i,j], C[i,j]
    U_out = np.zeros((4, 7), dtype=int)
    V_out = np.zeros((4, 7), dtype=int)
    W_out = np.zeros((4, 7), dtype=int)
    
    # M1: (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
    U_out[0, 0] = 1; U_out[3, 0] = 1
    V_out[0, 0] = 1; V_out[3, 0] = 1
    W_out[0, 0] = 1; W_out[3, 0] = 1
    
    # M2: (a10+a11)b00 -> c10 += M2, c11 -= M2
    U_out[2, 1] = 1; U_out[3, 1] = 1
    V_out[0, 1] = 1
    W_out[2, 1] = 1; W_out[3, 1] = -1
    
    # M3: a00(b10-b11) -> c01 += M3, c11 += M3
    U_out[0, 2] = 1
    V_out[2, 2] = 1; V_out[3, 2] = -1
    W_out[1, 2] = 1; W_out[3, 2] = 1
    
    # M4: a11(b00-b01) -> c00 += M4, c01 += M4
    U_out[3, 3] = 1
    V_out[0, 3] = 1; V_out[1, 3] = -1
    W_out[0, 3] = 1; W_out[1, 3] = 1
    
    # M5: (a00+a01)b11 -> c00 -= M5, c01 += M5
    U_out[0, 4] = 1; U_out[1, 4] = 1
    V_out[3, 4] = 1
    W_out[0, 4] = -1; W_out[1, 4] = 1
    
    # M6: (a10-a00)(b00+b01) -> c10 += M6
    U_out[2, 5] = 1; U_out[0, 5] = -1
    V_out[0, 5] = 1; V_out[1, 5] = 1
    W_out[2, 5] = 1
    
    # M7: (a01-a11)(b10+b11) -> c00 += M7
    U_out[1, 6] = 1; U_out[3, 6] = -1
    V_out[2, 6] = 1; V_out[3, 6] = 1
    W_out[0, 6] = 1
    
    # 3x3 Schoolbook decomposition
    # U, V, W have shape (9, 27)
    # Indexing: row index = i*3 + j for A[i,j], B[i,j], C[i,j]
    U_in = np.zeros((9, 27), dtype=int)
    V_in = np.zeros((9, 27), dtype=int)
    W_in = np.zeros((9, 27), dtype=int)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_in[i * 3 + j, col] = 1
                V_in[j * 3 + k, col] = 1
                W_in[k * 3 + i, col] = 1
                col += 1
    
    # Build the 6x6 decomposition
    # U: shape (36, 189), indexed by (block_i, block_j, sub_i, sub_j) -> (block_i*3 + sub_i)*6 + (block_j*3 + sub_j)
    # V: shape (36, 189), indexed by (block_j, block_k, sub_j, sub_k) -> (block_j*3 + sub_j)*6 + (block_k*3 + sub_k)
    # W: shape (36, 189), indexed by (block_k, block_i, sub_k, sub_i) -> (block_k*3 + sub_k)*6 + (block_i*3 + sub_i)
    #     Note: trace convention W[k*n+i, r] where k is column of C, i is row of C
    
    U = np.zeros((36, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    for c_out in range(7):
        for c_in in range(27):
            c = c_out * 27 + c_in
            
            # U: A[i,j] where i = block_i*3 + sub_i, j = block_j*3 + sub_j
            # U_out indexed by (block_i, block_j) -> block_i*2 + block_j
            # U_in indexed by (sub_i, sub_j) -> sub_i*3 + sub_j
            for r_out in range(4):
                if U_out[r_out, c_out] != 0:
                    block_i = r_out // 2
                    block_j = r_out % 2
                    val_out = U_out[r_out, c_out]
                    for r_in in range(9):
                        if U_in[r_in, c_in] != 0:
                            sub_i = r_in // 3
                            sub_j = r_in % 3
                            val_in = U_in[r_in, c_in]
                            i = block_i * 3 + sub_i
                            j = block_j * 3 + sub_j
                            U[i * 6 + j, c] += val_out * val_in
            
            # V: B[j,k] where j = block_j*3 + sub_j, k = block_k*3 + sub_k
            # V_out indexed by (block_j, block_k) -> block_j*2 + block_k
            # V_in indexed by (sub_j, sub_k) -> sub_j*3 + sub_k
            for r_out in range(4):
                if V_out[r_out, c_out] != 0:
                    block_j = r_out // 2
                    block_k = r_out % 2
                    val_out = V_out[r_out, c_out]
                    for r_in in range(9):
                        if V_in[r_in, c_in] != 0:
                            sub_j = r_in // 3
                            sub_k = r_in % 3
                            val_in = V_in[r_in, c_in]
                            j = block_j * 3 + sub_j
                            k = block_k * 3 + sub_k
                            V[j * 6 + k, c] += val_out * val_in
            
            # W: C[i,k] where i = block_i*3 + sub_i, k = block_k*3 + sub_k
            # Trace convention: W[k*n+i, r] -> index = k*6 + i
            # W_out indexed by (block_k, block_i) -> block_k*2 + block_i
            # W_in indexed by (sub_k, sub_i) -> sub_k*3 + sub_i
            for r_out in range(4):
                if W_out[r_out, c_out] != 0:
                    block_k = r_out // 2
                    block_i = r_out % 2
                    val_out = W_out[r_out, c_out]
                    for r_in in range(9):
                        if W_in[r_in, c_in] != 0:
                            sub_k = r_in // 3
                            sub_i = r_in % 3
                            val_in = W_in[r_in, c_in]
                            k = block_k * 3 + sub_k
                            i = block_i * 3 + sub_i
                            W[k * 6 + i, c] += val_out * val_in
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
