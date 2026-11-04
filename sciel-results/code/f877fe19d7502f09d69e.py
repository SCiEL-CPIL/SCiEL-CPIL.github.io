import numpy as np

DIMENSIONS = [4, 4, 6]

# Strassen's algorithm for 2x2x2 matrix multiplication
# Rank 7 decomposition
def _strassen_2x2():
    # U, V, W for 2x2x2
    # Indices: i,j in 0..1 for A; j,k in 0..1 for B; k,i in 0..1 for C
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:10, 3:11
    # C indices: 0:00, 1:01, 2:10, 3:11
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = -1; W[2, 0] = -1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[3, 5] = -1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Decompose into 4x4x4 and 4x4x2
    # 4x4x4 uses Strassen Kronecker (rank 49)
    # 4x4x2 uses Schoolbook (rank 32)
    
    U_str, V_str, W_str = _strassen_2x2()
    
    # Kronecker product for 4x4x4
    # A is 4x4, B is 4x4, C is 4x4
    # U_kron shape (16, 49)
    # V_kron shape (16, 49)
    # W_kron shape (16, 49)
    
    U_kron = np.zeros((16, 49), dtype=int)
    V_kron = np.zeros((16, 49), dtype=int)
    W_kron = np.zeros((16, 49), dtype=int)
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            # A index (i1*2+i2, j1*2+j2) -> (i1*2+i2)*4 + (j1*2+j2)
                            idx_A = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                            # B index (j1*2+j2, k1*2+k2) -> (j1*2+j2)*4 + (k1*2+k2)
                            # Wait, B is 4x4.
                            # V index in 2x2 is (j, k).
                            # In 4x4, B index is (j, k) where j,k in 0..3.
                            # j = j1*2 + j2? No.
                            # The Kronecker product structure:
                            # A = A11 A12 ...
                            # The indices in U_str are for 2x2 blocks.
                            # U_str[i1, j1] corresponds to block (i1, j1).
                            # Inside the block, indices are i2, j2.
                            # So U_kron[i, j] = U_str[i1, j1] * U_str[i2, j2]?
                            # No, U_str is for the whole 2x2 matrix.
                            # U_str index is i*2+j.
                            # So U_kron index is (i1*2+i2)*4 + (j1*2+j2).
                            # Value is U_str[i1*2+j1, r1] * U_str[i2*2+j2, r2].
                            
                            val = U_str[i1 * 2 + j1, r1] * U_str[i2 * 2 + j2, r2]
                            U_kron[idx_A, r] = val
                            
                            # V index: j, k
                            # j = j1*2 + j2? No, j is the column of A / row of B.
                            # In Kronecker, j = j1*2 + j2 is not correct.
                            # j is the index in 0..3.
                            # j = j1*2 + j2?
                            # If A is 4x4, A_{i,j}.
                            # i = i1*2 + i2. j = j1*2 + j2.
                            # B_{j,k}. j = j1*2 + j2. k = k1*2 + k2.
                            # V_str index is j*2+k.
                            # So V_kron index is j*4+k = (j1*2+j2)*4 + (k1*2+k2).
                            # Value is V_str[j1*2+k1, r1] * V_str[j2*2+k2, r2]?
                            # No, V_str is for 2x2 B.
                            # V_str index j*2+k.
                            # So V_kron[j, k] = V_str[j1, k1, r1] * V_str[j2, k2, r2]?
                            # Yes.
                            
                            for k1 in range(2):
                                for k2 in range(2):
                                    idx_B = (j1 * 2 + j2) * 4 + (k1 * 2 + k2)
                                    val_v = V_str[j1 * 2 + k1, r1] * V_str[j2 * 2 + k2, r2]
                                    V_kron[idx_B, r] = val_v
                                    
                                    # W index: k, i
                                    # W_str index k*2+i.
                                    # W_kron index k*4+i = (k1*2+k2)*4 + (i1*2+i2).
                                    idx_C = (k1 * 2 + k2) * 4 + (i1 * 2 + i2)
                                    val_w = W_str[k1 * 2 + i1, r1] * W_str[k2 * 2 + i2, r2]
                                    W_kron[idx_C, r] = val_w
    
    # 4x4x2 Schoolbook
    # A 4x4, B 4x2, C 4x2
    # Rank 32
    U_sb = np.zeros((16, 32), dtype=int)
    V_sb = np.zeros((8, 32), dtype=int)
    W_sb = np.zeros((8, 32), dtype=int)
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U_sb[i * 4 + j, col] = 1
                V_sb[j * 2 + k, col] = 1
                W_sb[k * 4 + i, col] = 1
                col += 1
    
    # Combine
    U_final = np.hstack([U_kron, U_sb])
    V_final = np.hstack([V_kron, V_sb])
    W_final = np.hstack([W_kron, W_sb])
    
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
