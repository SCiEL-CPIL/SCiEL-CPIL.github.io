"""Matrix multiplication 6x6x6 using 3x3 block decomposition with 2x2 Strassen (rank 189)."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = start_time + 140  # 20s safety margin
    
    n, m, p = 6, 6, 6
    nb = 3  # number of blocks
    bs = 2  # block size
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # U33[i*2+j, c], V33[j*2+k, c], W33[k*2+i, c]
    # Indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3 (k*n+i: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3)
    
    U33 = np.zeros((4, 7), dtype=np.float64)
    V33 = np.zeros((4, 7), dtype=np.float64)
    W33 = np.zeros((4, 7), dtype=np.float64)
    
    # M0
    U33[0, 0] = 1; U33[3, 0] = 1
    V33[0, 0] = 1; V33[3, 0] = 1
    W33[0, 0] = 1; W33[3, 0] = 1
    
    # M1
    U33[2, 1] = 1; U33[3, 1] = 1
    V33[0, 1] = 1
    W33[2, 1] = 1; W33[3, 1] = -1
    
    # M2
    U33[0, 2] = 1
    V33[1, 2] = 1; V33[3, 2] = -1
    W33[1, 2] = 1; W33[3, 2] = 1
    
    # M3
    U33[3, 3] = 1
    V33[2, 3] = 1; V33[0, 3] = -1
    W33[0, 3] = 1; W33[2, 3] = 1
    
    # M4
    U33[0, 4] = 1; U33[1, 4] = 1
    V33[3, 4] = 1
    W33[0, 4] = -1; W33[1, 4] = 1
    
    # M5
    U33[2, 5] = 1; U33[0, 5] = -1
    V33[0, 5] = 1; V33[1, 5] = 1
    W33[3, 5] = 1
    
    # M6
    U33[1, 6] = 1; U33[3, 6] = -1
    V33[2, 6] = 1; V33[3, 6] = 1
    W33[0, 6] = 1
    
    # Verify Strassen 2x2
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Term: U[i,j] V[j,k] W[k,i]
    verified = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U33[i*2+j, c] * V33[j*2+k, c] * W33[k*2+i, c]
                if s != 1:
                    verified = False
                    break
            if not verified: break
        if not verified: break
    
    if not verified:
        # Fallback to schoolbook 6x6
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Lift to 6x6 using 3x3 blocks of 2x2
    # Rank = 3 * 3 * 3 * 7 = 189
    total_rank = 189
    U = np.zeros((36, total_rank), dtype=np.float64)
    V = np.zeros((36, total_rank), dtype=np.float64)
    W = np.zeros((36, total_rank), dtype=np.float64)
    
    col = 0
    for bi in range(3):
        for bk in range(3):
            for bj in range(3):
                for c in range(7):
                    # U part: A block (bi, bj)
                    for ii in range(2):
                        for jj in range(2):
                            u_idx_22 = ii*2+jj
                            val = U33[u_idx_22, c]
                            if val != 0:
                                u_idx_66 = (bi*2+ii)*6 + (bj*2+jj)
                                U[u_idx_66, col] = val
                    # V part: B block (bj, bk)
                    for jj in range(2):
                        for kk in range(2):
                            v_idx_22 = jj*2+kk
                            val = V33[v_idx_22, c]
                            if val != 0:
                                v_idx_66 = (bj*2+jj)*6 + (bk*2+kk)
                                V[v_idx_66, col] = val
                    # W part: C block (bi, bk)
                    for kk in range(2):
                        for ii in range(2):
                            w_idx_22 = kk*2+ii
                            val = W33[w_idx_22, c]
                            if val != 0:
                                w_idx_66 = (bi*2+ii)*6 + (bk*2+kk)
                                W[w_idx_66, col] = val
                    col += 1
    
    # Quick verification of the full 6x6 decomposition
    # Pick random small matrices
    # A = np.random.rand(6, 6)
    # B = np.random.rand(6, 6)
    # C_expected = A @ B
    # C_decomp = np.zeros((6, 6))
    # for r in range(total_rank):
    #     for i in range(6):
    #         for j in range(6):
    #             if U[i*6+j, r] != 0:
    #                 for k in range(6):
    #                     if V[j*6+k, r] != 0 and W[k*6+i, r] != 0:
    #                         C_decomp[i, k] += U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r] * A[i, j] * B[j, k]
    # if not np.allclose(C_expected, C_decomp):
    #     # Fallback
    #     pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
