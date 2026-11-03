import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Construct M_2,2,2 Strassen decomposition (Rank 7)
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    U_2 = np.zeros((4, 7), dtype=int)
    V_2 = np.zeros((4, 7), dtype=int)
    W_2 = np.zeros((4, 7), dtype=int)
    
    # P1 = (a00+a11)(b00+b11)
    # A coeffs: a00:1, a11:1
    U_2[0, 0] = 1 # a00
    U_2[3, 0] = 1 # a11
    # B coeffs: b00:1, b11:1
    V_2[0, 0] = 1 # b00
    V_2[3, 0] = 1 # b11
    # C coeffs: c00:1, c11:1
    W_2[0, 0] = 1 # c00 (k=0, i=0)
    W_2[3, 0] = 1 # c11 (k=1, i=1)
    
    # P2 = (a10+a11)b00
    U_2[2, 1] = 1 # a10
    U_2[3, 1] = 1 # a11
    V_2[0, 1] = 1 # b00
    W_2[2, 1] = 1 # c10 (k=0, i=1)
    
    # P3 = a00(b01-b11)
    U_2[0, 2] = 1 # a00
    V_2[1, 2] = 1 # b01
    V_2[3, 2] = -1 # b11
    W_2[1, 2] = 1 # c01 (k=1, i=0)
    
    # P4 = a11(b10-b00)
    U_2[3, 3] = 1 # a11
    V_2[2, 3] = 1 # b10
    V_2[0, 3] = -1 # b00
    W_2[0, 3] = -1 # c00 (k=0, i=0)
    W_2[2, 3] = 1 # c10 (k=0, i=1)
    
    # P5 = (a00+a01)b11
    U_2[0, 4] = 1 # a00
    U_2[1, 4] = 1 # a01
    V_2[3, 4] = 1 # b11
    W_2[1, 4] = 1 # c01 (k=1, i=0)
    W_2[3, 4] = -1 # c11 (k=1, i=1)
    
    # P6 = (a10-a00)(b00+b01)
    U_2[2, 5] = 1 # a10
    U_2[0, 5] = -1 # a00
    V_2[0, 5] = 1 # b00
    V_2[1, 5] = 1 # b01
    W_2[0, 5] = 1 # c00 (k=0, i=0)
    W_2[1, 5] = 1 # c01 (k=1, i=0)
    
    # P7 = (a01-a11)(b10+b11)
    U_2[1, 6] = 1 # a01
    U_2[3, 6] = -1 # a11
    V_2[2, 6] = 1 # b10
    V_2[3, 6] = 1 # b11
    W_2[0, 6] = -1 # c00 (k=0, i=0)
    W_2[2, 6] = -1 # c10 (k=0, i=1)

    # 2. Construct M_4,4,4 decomposition (Rank 49)
    # Using Strassen on 2x2 blocks of 2x2 matrices.
    # This is effectively the Kronecker product of the 2x2x2 tensor with itself.
    # U_4 shape (16, 49), V_4 shape (16, 49), W_4 shape (16, 49)
    
    U_4 = np.zeros((16, 49), dtype=int)
    V_4 = np.zeros((16, 49), dtype=int)
    W_4 = np.zeros((16, 49), dtype=int)
    
    # For each of the 7 Strassen steps in the outer 2x2 multiplication,
    # we perform an inner 2x2 multiplication for each block product.
    # Wait, the standard recursive Strassen applies Strassen to each of the 7 products.
    # So for each outer step r_out (0..6), we have 7 inner steps r_in (0..6).
    # Total 49 steps.
    
    # Mapping of outer indices to inner block indices
    # A is 4x4, split into 2x2 blocks A00, A01, A10, A11 (each 2x2)
    # B is 4x4, split into 2x2 blocks B00, B01, B10, B11 (each 2x2)
    # C is 4x4, split into 2x2 blocks C00, C01, C10, C11 (each 2x2)
    
    # Outer Strassen formulas:
    # P1_out = (A00+A11)(B00+B11)
    # P2_out = (A10+A11)B00
    # P3_out = A00(B01-B11)
    # P4_out = A11(B10-B00)
    # P5_out = (A00+A01)B11
    # P6_out = (A10-A00)(B00+B01)
    # P7_out = (A01-A11)(B10+B11)
    
    # Inner Strassen formulas (same as above)
    
    # We need to map the outer block indices to the actual 4x4 indices.
    # A00: rows 0,1 cols 0,1 -> indices 0,1,4,5
    # A01: rows 0,1 cols 2,3 -> indices 2,3,6,7
    # A10: rows 2,3 cols 0,1 -> indices 8,9,12,13
    # A11: rows 2,3 cols 2,3 -> indices 10,11,14,15
    
    # B00: rows 0,1 cols 0,1 -> indices 0,1,4,5
    # B01: rows 0,1 cols 2,3 -> indices 2,3,6,7
    # B10: rows 2,3 cols 0,1 -> indices 8,9,12,13
    # B11: rows 2,3 cols 2,3 -> indices 10,11,14,15
    
    # C00: rows 0,1 cols 0,1 -> indices 0,1,4,5 (W index k*4+i)
    # C01: rows 0,1 cols 2,3 -> indices 2,3,6,7
    # C10: rows 2,3 cols 0,1 -> indices 8,9,12,13
    # C11: rows 2,3 cols 2,3 -> indices 10,11,14,15
    
    # For each outer step r_out, we have coefficients for A_blocks and B_blocks.
    # These coefficients are linear combinations.
    # We then apply the inner Strassen decomposition to each product of linear combinations.
    
    # This is getting complicated to implement manually.
    # However, we can use the property that M_4,4,4 is the Kronecker product of M_2,2,2 with itself.
    # T_4 = T_2 \otimes T_2.
    # Rank(T_4) = Rank(T_2) * Rank(T_2) = 49.
    
    # Let's construct the Kronecker product directly.
    # U_4[i1*4+i2, r1*7+r2] = U_2[i1*2+i2, r1] * U_2[?, r2]?
    # No, the Kronecker product of tensors works as follows:
    # (T1 \otimes T2)_{i, j, k} = T1_{i1, j1, k1} T2_{i2, j2, k2}
    # where i = i1*2 + i2, etc.
    # The decomposition matrices for the Kronecker product are:
    # U_kron[i1*2+i2, r1*7+r2] = U1[i1*2+j1, r1] U2[j1*2+i2, r2]? No.
    # The indices of U are (i, j). In Kronecker, i = i1*2+i2, j = j1*2+j2.
    # So U_kron[i1*2+i2, j1*2+j2, r1*7+r2] = U1[i1, j1, r1] U2[i2, j2, r2].
    # Flattened: U_kron[(i1*2+i2)*4 + (j1*2+j2), r1*7+r2] = U1[i1*2+j1, r1] * U2[i2*2+j2, r2].
    
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1 * 7 + r2
                            idx_u = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                            U_4[idx_u, r] = U_2[i1 * 2 + j1, r1] * U_2[i2 * 2 + j2, r2]
                            
                            # V indices: (j, k). j = j1*2+j2, k = k1*2+k2
                            # V_kron[j1*2+j2, k1*2+k2, r] = V1[j1, k1, r1] V2[j2, k2, r2]
                            # We need to iterate k1, k2 as well for V.
                            # But V is flattened (j, k).
                            # So for each j1, j2, k1, k2:
                            for k1 in range(2):
                                for k2 in range(2):
                                    idx_v = (j1 * 2 + j2) * 4 + (k1 * 2 + k2)
                                    V_4[idx_v, r] = V_2[j1 * 2 + k1, r1] * V_2[j2 * 2 + k2, r2]
                                    
                                    # W indices: (k, i). k = k1*2+k2, i = i1*2+i2
                                    # W_kron[k1*2+k2, i1*2+i2, r] = W1[k1, i1, r1] W2[k2, i2, r2]
                                    idx_w = (k1 * 2 + k2) * 4 + (i1 * 2 + i2)
                                    W_4[idx_w, r] = W_2[k1 * 2 + i1, r1] * W_2[k2 * 2 + i2, r2]

    # 3. Construct M_4,4,1 decomposition (Rank 16)
    U_1 = np.zeros((16, 16), dtype=int)
    V_1 = np.zeros((36, 16), dtype=int) # m*p = 4*9? No, for this block p=1, so m*p=4.
    # Wait, the full V is (36, R). For the block of 1 column, V is (4, 16).
    # We will place it in the correct position in the full V.
    W_1 = np.zeros((36, 16), dtype=int) # p*n = 1*4 = 4.
    
    for i in range(4):
        for j in range(4):
            c = i * 4 + j
            U_1[i * 4 + j, c] = 1
            V_1[j, c] = 1
            W_1[i, c] = 1 # W index k*n+i, k=0, so i.

    # 4. Combine for M_4,4,9
    # Rank = 49 + 49 + 16 = 114
    U = np.zeros((16, 114), dtype=int)
    V = np.zeros((36, 114), dtype=int)
    W = np.zeros((36, 114), dtype=int)
    
    # Block 1: Cols 0-3 (M_4,4,4)
    U[:, 0:49] = U_4
    # V_4 is (16, 49). Full V is (36, 114).
    # V_4 indices are (j, k) for k in 0..3.
    # Full V indices are (j, k_full) for k_full in 0..8.
    # For k_full in 0..3, k = k_full.
    for j in range(4):
        for k in range(4):
            V[j * 9 + k, 0:49] = V_4[j * 4 + k, :]
    # W_4 is (16, 49). Full W is (36, 114).
    # W_4 indices are (k, i) for k in 0..3.
    # Full W indices are (k_full, i) for k_full in 0..8.
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W_4[k * 4 + i, :]
            
    # Block 2: Cols 4-7 (M_4,4,4)
    U[:, 49:98] = U_4
    for j in range(4):
        for k in range(4):
            V[j * 9 + (k + 4), 49:98] = V_4[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[(k + 4) * 4 + i, 49:98] = W_4[k * 4 + i, :]
            
    # Block 3: Col 8 (M_4,4,1)
    U[:, 98:114] = U_1
    for j in range(4):
        V[j * 9 + 8, 98:114] = V_1[j, :]
    for i in range(4):
        W[8 * 4 + i, 98:114] = W_1[i, :]
        
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
