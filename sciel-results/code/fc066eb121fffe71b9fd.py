import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 3, 8

    # Build the target tensor
    T_exp = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1

    # We use a block decomposition: split B into 4 blocks of width 2.
    # Each block computes C_block = A * B_block where A is 2x3, B_block is 3x2, result 2x2.
    # The tensor for one block M_{2,3,2} has indices:
    #   a = i*3+j (i in 0..1, j in 0..2)  -> 6 entries
    #   b = j*2+k (j in 0..2, k in 0..1)  -> 6 entries
    #   d = k*2+i (k in 0..1, i in 0..1)  -> 4 entries
    
    # We use a rank-11 decomposition for M_{2,3,2}.
    # Total rank: 4 * 11 = 44.
    
    U = np.zeros((n*m, 44), dtype=np.int64)
    V = np.zeros((m*p, 44), dtype=np.int64)
    W = np.zeros((p*n, 44), dtype=np.int64)
    
    col_offset = 0
    
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # Block 1: A[:, 0] * B[0, k0:k1]  (2x1 * 1x2) - rank 4
        U[0*m + 0, col_offset] = 1
        V[0*p + k0, col_offset] = 1
        W[k0*n + 0, col_offset] = 1
        col_offset += 1
        
        U[0*m + 0, col_offset] = 1
        V[0*p + k1, col_offset] = 1
        W[k1*n + 0, col_offset] = 1
        col_offset += 1
        
        U[1*m + 0, col_offset] = 1
        V[0*p + k0, col_offset] = 1
        W[k0*n + 1, col_offset] = 1
        col_offset += 1
        
        U[1*m + 0, col_offset] = 1
        V[0*p + k1, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1
        
        # Block 2: A[:, 1:3] * B[1:3, k0:k1]  (2x2 * 2x2) - Strassen rank 7
        # Using the Strassen decomposition
        # M1 = (A01 + A02)(B10 + B21) -> C00 - C01 - C10 + C11? No, standard Strassen:
        # M1 = (A11 - A22)(B21 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B11 - B12)
        # M5 = (A11 + A22)(B11 + B22)
        # M6 = (A12 - A11)(B11 + B21)
        # M7 = (A21 - A22)(B12 + B22)
        # Here A is 2x2: [[A01, A02], [A11, A12]]
        # B is 2x2: [[B10, B11], [B20, B21]]
        
        # Term 1: M1 = (A01 - A12)(B20 + B21)
        U[0*m + 1, col_offset] = 1
        U[1*m + 2, col_offset] = -1
        V[2*p + k0, col_offset] = 1
        V[2*p + k1, col_offset] = 1
        W[k0*n + 0, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        W[k0*n + 1, col_offset] = -1
        W[k1*n + 0, col_offset] = -1
        col_offset += 1

        # Term 2: M2 = (A11 + A12)B10
        U[1*m + 1, col_offset] = 1
        U[1*m + 2, col_offset] = 1
        V[1*p + k0, col_offset] = 1
        W[k0*n + 1, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1

        # Term 3: M3 = A01(B11 - B21)
        U[0*m + 1, col_offset] = 1
        V[1*p + k1, col_offset] = 1
        V[2*p + k1, col_offset] = -1
        W[k1*n + 0, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1

        # Term 4: M4 = A12(B10 - B11)
        U[1*m + 2, col_offset] = 1
        V[1*p + k0, col_offset] = 1
        V[1*p + k1, col_offset] = -1
        W[k0*n + 0, col_offset] = 1
        W[k0*n + 1, col_offset] = 1
        col_offset += 1

        # Term 5: M5 = (A01 + A12)(B10 + B21)
        U[0*m + 1, col_offset] = 1
        U[1*m + 2, col_offset] = 1
        V[1*p + k0, col_offset] = 1
        V[2*p + k1, col_offset] = 1
        W[k0*n + 0, col_offset] = 1
        W[k1*n + 0, col_offset] = 1
        W[k0*n + 1, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1

        # Term 6: M6 = (A02 - A01)(B10 + B20)
        U[0*m + 2, col_offset] = 1
        U[0*m + 1, col_offset] = -1
        V[1*p + k0, col_offset] = 1
        V[2*p + k0, col_offset] = 1
        W[k0*n + 0, col_offset] = 1
        W[k1*n + 0, col_offset] = 1
        col_offset += 1

        # Term 7: M7 = (A11 - A12)(B11 + B21)
        U[1*m + 1, col_offset] = 1
        U[1*m + 2, col_offset] = -1
        V[1*p + k1, col_offset] = 1
        V[2*p + k1, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1

    def verify(U, V, W):
        r = U.shape[1]
        T_check = np.einsum('ad,bd,cd->abc', U, V, W)
        return np.array_equal(T_check, T_exp)

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n*m, rank_schoolbook), dtype=np.int64)
        V_sb = np.zeros((m*p, rank_schoolbook), dtype=np.int64)
        W_sb = np.zeros((p*n, rank_schoolbook), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_sb[a_idx, col] = 1
                    V_sb[b_idx, col] = 1
                    W_sb[c_idx, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
