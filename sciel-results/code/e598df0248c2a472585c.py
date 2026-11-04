import time
import numpy as np

DIMENSIONS = [6, 6, 6]
DEADLINE = time.time() + 140

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition 6x6 into 2x2 blocks of 3x3 submatrices
    # Outer: 2x2 Strassen (rank 7), Inner: 3x3 schoolbook (rank 9)
    # Total rank = 7 * 9 = 63
    
    # 2x2 Strassen decomposition
    # U_out, V_out, W_out shapes (4, 7)
    # For 2x2: indices are (row, col) where row,col in {0,1}
    # U: input A partitioned as A00, A01, A10, A11 -> indices 0,1,2,3
    # V: input B partitioned as B00, B01, B10, B11 -> indices 0,1,2,3
    # W: output C partitioned as C00, C01, C10, C11 -> indices 0,1,2,3
    
    U_out = np.zeros((4, 7), dtype=int)
    V_out = np.zeros((4, 7), dtype=int)
    W_out = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    U_out[0, 0] = 1; U_out[3, 0] = 1
    V_out[0, 0] = 1; V_out[3, 0] = 1
    W_out[0, 0] = 1; W_out[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U_out[2, 1] = 1; U_out[3, 1] = 1
    V_out[0, 1] = 1
    W_out[1, 1] = 1; W_out[3, 1] = -1
    
    # M3 = A00(B10-B11)
    U_out[0, 2] = 1
    V_out[2, 2] = 1; V_out[3, 2] = -1
    W_out[2, 2] = 1; W_out[3, 2] = 1
    
    # M4 = A11(B00-B01)
    U_out[3, 3] = 1
    V_out[0, 3] = 1; V_out[1, 3] = -1
    W_out[0, 3] = 1; W_out[1, 3] = 1
    
    # M5 = (A00+A01)B11
    U_out[0, 4] = 1; U_out[1, 4] = 1
    V_out[3, 4] = 1
    W_out[0, 4] = -1; W_out[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U_out[2, 5] = 1; U_out[0, 5] = -1
    V_out[0, 5] = 1; V_out[1, 5] = 1
    W_out[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U_out[1, 6] = 1; U_out[3, 6] = -1
    V_out[2, 6] = 1; V_out[3, 6] = 1
    W_out[0, 6] = 1
    
    # 3x3 Schoolbook decomposition
    # U_in, V_in, W_in shapes (9, 9)
    U_in = np.zeros((9, 9), dtype=int)
    V_in = np.zeros((9, 9), dtype=int)
    W_in = np.zeros((9, 9), dtype=int)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_in[i * 3 + j, col] = 1
                V_in[j * 3 + k, col] = 1
                W_in[k * 3 + i, col] = 1
                col += 1
                
    rank = 7 * 9
    U = np.zeros((36, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    for c_out in range(7):
        for c_in in range(9):
            if time.time() > DEADLINE:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
            c = c_out * 9 + c_in
            
            # U: row index for A block (i_out, j_out) and sub-block (i_in, j_in)
            # Global row = (i_out * 3 + i_in) * 6 + (j_out * 3 + j_in)
            for r_out in range(4):
                if U_out[r_out, c_out] != 0:
                    i_out = r_out // 2
                    j_out = r_out % 2
                    val_out = U_out[r_out, c_out]
                    for r_in in range(9):
                        if U_in[r_in, c_in] != 0:
                            i_in = r_in // 3
                            j_in = r_in % 3
                            val_in = U_in[r_in, c_in]
                            r = (i_out * 3 + i_in) * 6 + (j_out * 3 + j_in)
                            U[r, c] += val_out * val_in
                            
            # V: row index for B block (j_out, k_out) and sub-block (j_in, k_in)
            # Global row = (j_out * 3 + j_in) * 6 + (k_out * 3 + k_in)
            for r_out in range(4):
                if V_out[r_out, c_out] != 0:
                    j_out = r_out // 2
                    k_out = r_out % 2
                    val_out = V_out[r_out, c_out]
                    for r_in in range(9):
                        if V_in[r_in, c_in] != 0:
                            j_in = r_in // 3
                            k_in = r_in % 3
                            val_in = V_in[r_in, c_in]
                            r = (j_out * 3 + j_in) * 6 + (k_out * 3 + k_in)
                            V[r, c] += val_out * val_in
                            
            # W: row index for C block (k_out, i_out) and sub-block (k_in, i_in)
            # Global row = (k_out * 3 + k_in) * 6 + (i_out * 3 + i_in)
            for r_out in range(4):
                if W_out[r_out, c_out] != 0:
                    k_out = r_out // 2
                    i_out = r_out % 2
                    val_out = W_out[r_out, c_out]
                    for r_in in range(9):
                        if W_in[r_in, c_in] != 0:
                            k_in = r_in // 3
                            i_in = r_in % 3
                            val_in = W_in[r_in, c_in]
                            r = (k_out * 3 + k_in) * 6 + (i_out * 3 + i_in)
                            W[r, c] += val_out * val_in
                            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
