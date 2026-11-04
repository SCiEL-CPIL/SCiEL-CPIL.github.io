import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 3, 4, 6
    
    # Rank 66 decomposition
    # U: (n*m, rank) = (12, 66)
    # V: (m*p, rank) = (24, 66)
    # W: (p*n, rank) = (18, 66)
    
    U = np.zeros((n * m, 66))
    V = np.zeros((m * p, 66))
    W = np.zeros((p * n, 66))
    
    col = 0
    
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # A indices
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        # B indices
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        # C indices (W index: k*n + i)
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)
        
        # Strassen 7 terms
        # 1
        U[A00, col] += 1; U[A11, col] += 1
        V[B00, col] += 1; V[B11, col] += 1
        W[C00, col] += 1; W[C11, col] += 1
        col += 1
        # 2
        U[A10, col] += 1; U[A11, col] += 1
        V[B00, col] += 1
        W[C10, col] += 1; W[C11, col] -= 1
        col += 1
        # 3
        U[A00, col] += 1
        V[B01, col] += 1; V[B11, col] -= 1
        W[C01, col] += 1; W[C11, col] += 1
        col += 1
        # 4
        U[A11, col] += 1
        V[B10, col] += 1; V[B00, col] -= 1
        W[C00, col] += 1; W[C10, col] += 1
        col += 1
        # 5
        U[A00, col] += 1; U[A01, col] += 1
        V[B11, col] += 1
        W[C00, col] -= 1; W[C01, col] += 1
        col += 1
        # 6
        U[A10, col] += 1; U[A00, col] -= 1
        V[B00, col] += 1; V[B01, col] += 1
        W[C11, col] += 1
        col += 1
        # 7
        U[A01, col] += 1; U[A11, col] -= 1
        V[B10, col] += 1; V[B11, col] += 1
        W[C00, col] += 1
        col += 1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # A: 1x2 (row a_r, cols a_c, a_c+1)
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        # B: 2x2 (rows b_r, b_r+1; cols b_c, b_c+1)
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        # C: 1x2 (row c_r, cols c_c, c_c+1)
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r
        
        # 4 terms
        U[A0, col] += 1; V[B00, col] += 1; W[C0, col] += 1
        col += 1
        U[A1, col] += 1; V[B10, col] += 1; W[C0, col] += 1
        col += 1
        U[A0, col] += 1; V[B01, col] += 1; W[C1, col] += 1
        col += 1
        U[A1, col] += 1; V[B11, col] += 1; W[C1, col] += 1
        col += 1

    # Row 0 of A (1x4)
    # Split A into (1x2, 1x2)
    # Split B into (2x2, 2x2) blocks for cols 0-1, 2-3, 4-5
    # C cols 0-1
    add_1x2_2x2(0, 0, 0, 0, 0, 0) # A[0,0:2] * B[0:2, 0:2]
    add_1x2_2x2(0, 2, 2, 0, 0, 0) # A[0,2:4] * B[2:4, 0:2]
    # C cols 2-3
    add_1x2_2x2(0, 0, 0, 2, 0, 2) # A[0,0:2] * B[0:2, 2:4]
    add_1x2_2x2(0, 2, 2, 2, 0, 2) # A[0,2:4] * B[2:4, 2:4]
    # C cols 4-5
    add_1x2_2x2(0, 0, 0, 4, 0, 4) # A[0,0:2] * B[0:2, 4:6]
    add_1x2_2x2(0, 2, 2, 4, 0, 4) # A[0,2:4] * B[2:4, 4:6]
    
    # Rows 1-2 of A (2x4)
    # Split A into (2x2, 2x2)
    # Split B into (2x2, 2x2) blocks
    # C rows 1-2, cols 0-1
    add_strassen(1, 0, 0, 0, 1, 0) # A[1:3, 0:2] * B[0:2, 0:2]
    add_strassen(1, 2, 2, 0, 1, 0) # A[1:3, 2:4] * B[2:4, 0:2]
    # C rows 1-2, cols 2-3
    add_strassen(1, 0, 0, 2, 1, 2) # A[1:3, 0:2] * B[0:2, 2:4]
    add_strassen(1, 2, 2, 2, 1, 2) # A[1:3, 2:4] * B[2:4, 2:4]
    # C rows 1-2, cols 4-5
    add_strassen(1, 0, 0, 4, 1, 4) # A[1:3, 0:2] * B[0:2, 4:6]
    add_strassen(1, 2, 2, 4, 1, 4) # A[1:3, 2:4] * B[2:4, 4:6]
    
    # Verification
    def verify(U, V, W):
        r = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    expected = 1.0
                    max_err = max(max_err, abs(s - expected))
        return max_err

    err = verify(U, V, W)
    if err > 1e-10:
        # Fallback to schoolbook if something went wrong (should not happen)
        rank_fb = n * m * p
        U = np.zeros((n * m, rank_fb))
        V = np.zeros((m * p, rank_fb))
        W = np.zeros((p * n, rank_fb))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1
                    V[j*p+k, idx] = 1
                    W[k*n+i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
