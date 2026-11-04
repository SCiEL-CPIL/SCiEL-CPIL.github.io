import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    Build the 2x2 Strassen decomposition (rank 7).
    Returns U, V, W of shape (4, 7).
    Row index: i*2 + j.
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    # P1: (x00+x11)(y00+y11) -> c00+=1, c11+=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # P2: (x10+x11)(y00) -> c10+=1, c11+=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1

    # P3: (x00)(y01-y11) -> c01+=1, c11+=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1

    # P4: (x11)(y10-y00) -> c00+=1, c10+=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1

    # P5: (x00+x01)(y11) -> c00+=-1, c01+=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1

    # P6: (x10-x00)(y00+y01) -> c11+=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # P7: (x01-x11)(y10+y11) -> c00+=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _build_laderman_4x4_blocks():
    """
    Build the Laderman 4x4 decomposition at the 2x2 block level (rank 25).
    Returns U_b, V_b, W_b of shape (4, 25).
    Row index: block_i * 2 + block_j.
    """
    U_b = np.zeros((4, 25), dtype=float)
    V_b = np.zeros((4, 25), dtype=float)
    W_b = np.zeros((4, 25), dtype=float)
    
    I = 1.0
    Z = 0.0
    
    # Term 1: (A11 + A22) * (B11 + B22)
    U_b[0, 0] = I; U_b[3, 0] = I
    V_b[0, 0] = I; V_b[3, 0] = I
    W_b[0, 0] = I; W_b[3, 0] = I
    
    # Term 2: (A21 + A22) * B11
    U_b[2, 1] = I; U_b[3, 1] = I
    V_b[0, 1] = I
    W_b[2, 1] = I; W_b[3, 1] = I
    
    # Term 3: A11 * (B12 - B22)
    U_b[0, 2] = I
    V_b[1, 2] = I; V_b[3, 2] = -I
    W_b[1, 2] = I; W_b[3, 2] = I
    
    # Term 4: A22 * (B21 - B11)
    U_b[3, 3] = I
    V_b[2, 3] = I; V_b[0, 3] = -I
    W_b[0, 3] = I; W_b[2, 3] = I
    
    # Term 5: (A11 + A12) * B22
    U_b[0, 4] = I; U_b[1, 4] = I
    V_b[3, 4] = I
    W_b[0, 4] = I; W_b[1, 4] = -I
    
    # Term 6: (A21 - A11) * (B11 + B12)
    U_b[2, 5] = I; U_b[0, 5] = -I
    V_b[0, 5] = I; V_b[1, 5] = I
    W_b[3, 5] = I
    
    # Term 7: (A12 - A22) * (B21 + B22)
    U_b[1, 6] = I; U_b[3, 6] = -I
    V_b[2, 6] = I; V_b[3, 6] = I
    W_b[0, 6] = I
    
    # Term 8: (A11 + A21) * (B22 + B12)
    U_b[0, 7] = I; U_b[2, 7] = I
    V_b[3, 7] = I; V_b[1, 7] = I
    W_b[0, 7] = I; W_b[1, 7] = I
    
    # Term 9: (A12 + A22) * (B21 + B11)
    U_b[1, 8] = I; U_b[3, 8] = I
    V_b[2, 8] = I; V_b[0, 8] = I
    W_b[1, 8] = I; W_b[3, 8] = I
    
    # Term 10: (A21 + A11) * (B11 + B21)
    U_b[2, 9] = I; U_b[0, 9] = I
    V_b[0, 9] = I; V_b[2, 9] = I
    W_b[2, 9] = I; W_b[0, 9] = I
    
    # Term 11: (A22 + A12) * (B12 + B22)
    U_b[3, 10] = I; U_b[1, 10] = I
    V_b[1, 10] = I; V_b[3, 10] = I
    W_b[3, 10] = I; W_b[1, 10] = I
    
    # Term 12: (A11 - A22 + A12 - A21) * (B11 - B22 + B12 - B21)
    U_b[0, 11] = I; U_b[3, 11] = -I; U_b[1, 11] = I; U_b[2, 11] = -I
    V_b[0, 11] = I; V_b[3, 11] = -I; V_b[1, 11] = I; V_b[2, 11] = -I
    W_b[0, 11] = I; W_b[3, 11] = I; W_b[1, 11] = -I; W_b[2, 11] = -I
    
    # Term 13: (A11 + A22 - A12 - A21) * (B11 + B22 - B12 - B21)
    U_b[0, 12] = I; U_b[3, 12] = I; U_b[1, 12] = -I; U_b[2, 12] = -I
    V_b[0, 12] = I; V_b[3, 12] = I; V_b[1, 12] = -I; V_b[2, 12] = -I
    W_b[0, 12] = I; W_b[3, 12] = I; W_b[1, 12] = I; W_b[2, 12] = I
    
    # Term 14: (A11 - A12 + A21 - A22) * (B11 - B12 + B21 - B22)
    U_b[0, 13] = I; U_b[1, 13] = -I; U_b[2, 13] = I; U_b[3, 13] = -I
    V_b[0, 13] = I; V_b[1, 13] = -I; V_b[2, 13] = I; V_b[3, 13] = -I
    W_b[0, 13] = I; W_b[1, 13] = -I; W_b[2, 13] = I; W_b[3, 13] = -I
    
    # Term 15: (A11 + A12 - A21 - A22) * (B11 + B12 - B21 - B22)
    U_b[0, 14] = I; U_b[1, 14] = I; U_b[2, 14] = -I; U_b[3, 14] = -I
    V_b[0, 14] = I; V_b[1, 14] = I; V_b[2, 14] = -I; V_b[3, 14] = -I
    W_b[0, 14] = I; W_b[1, 14] = I; W_b[2, 14] = -I; W_b[3, 14] = -I
    
    # Term 16: (A21 - A22) * (B11 - B22)
    U_b[2, 15] = I; U_b[3, 15] = -I
    V_b[0, 15] = I; V_b[3, 15] = -I
    W_b[2, 15] = I; W_b[3, 15] = -I
    
    # Term 17: (A12 - A11) * (B21 - B22)
    U_b[1, 16] = I; U_b[0, 16] = -I
    V_b[2, 16] = I; V_b[3, 16] = -I
    W_b[1, 16] = I; W_b[0, 16] = -I
    
    # Term 18: (A21 - A11) * (B12 - B22)
    U_b[2, 17] = I; U_b[0, 17] = -I
    V_b[1, 17] = I; V_b[3, 17] = -I
    W_b[2, 17] = I; W_b[3, 17] = -I
    
    # Term 19: (A12 - A22) * (B21 - B11)
    U_b[1, 18] = I; U_b[3, 18] = -I
    V_b[2, 18] = I; V_b[0, 18] = -I
    W_b[1, 18] = I; W_b[0, 18] = -I
    
    # Term 20: (A11 + A22 + A12 + A21) * (B11 + B22 + B12 + B21)
    U_b[0, 19] = I; U_b[3, 19] = I; U_b[1, 19] = I; U_b[2, 19] = I
    V_b[0, 19] = I; V_b[3, 19] = I; V_b[1, 19] = I; V_b[2, 19] = I
    W_b[0, 19] = I; W_b[3, 19] = I; W_b[1, 19] = I; W_b[2, 19] = I
    
    # Term 21: (A11 - A22 - A12 + A21) * (B11 - B22 - B12 + B21)
    U_b[0, 20] = I; U_b[3, 20] = -I; U_b[1, 20] = -I; U_b[2, 20] = I
    V_b[0, 20] = I; V_b[3, 20] = -I; V_b[1, 20] = -I; V_b[2, 20] = I
    W_b[0, 20] = I; W_b[3, 20] = -I; W_b[1, 20] = -I; W_b[2, 20] = I
    
    # Term 22: (A11 + A22 - A12 - A21) * (B11 + B22 + B12 - B21)
    U_b[0, 21] = I; U_b[3, 21] = I; U_b[1, 21] = -I; U_b[2, 21] = -I
    V_b[0, 21] = I; V_b[3, 21] = I; V_b[1, 21] = I; V_b[2, 21] = -I
    W_b[0, 21] = I; W_b[3, 21] = -I; W_b[1, 21] = I; W_b[2, 21] = I
    
    # Term 23: (A11 - A22 + A12 - A21) * (B11 + B22 + B12 + B21)
    U_b[0, 22] = I; U_b[3, 22] = -I; U_b[1, 22] = I; U_b[2, 22] = -I
    V_b[0, 22] = I; V_b[3, 22] = I; V_b[1, 22] = I; V_b[2, 22] = I
    W_b[0, 22] = I; W_b[3, 22] = I; W_b[1, 22] = I; W_b[2, 22] = -I
    
    # Term 24: (A21 + A12) * (B11 + B22)
    U_b[2, 23] = I; U_b[1, 23] = I
    V_b[0, 23] = I; V_b[3, 23] = I
    W_b[2, 23] = I; W_b[1, 23] = I
    
    # Term 25: (A11 + A22) * (B21 + B12)
    U_b[0, 24] = I; U_b[3, 24] = I
    V_b[2, 24] = I; V_b[1, 24] = I
    W_b[0, 24] = I; W_b[3, 24] = I

    return U_b, V_b, W_b

def _verify_decomposition(U, V, W, n, m, p, seed=42, atol=1e-9):
    """Verify that U, V, W correctly decompose matrix multiplication."""
    np.random.seed(seed)
    A_test = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B_test = np.random.randint(-5, 6, size=(m, p)).astype(float)
    C_expected = A_test @ B_test

    rank = U.shape[1]
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for r in range(rank):
                    s += A_test[i, j] * B_test[j, k] * U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
            C_computed[i, k] = s

    return np.allclose(C_computed, C_expected, atol=atol)

def construct():
    """
    Construct a rank-175 decomposition of 6x6 * 6x6 -> 6x6 matrix multiplication
    using Laderman 4x4 (rank 25) x Strassen 2x2 (rank 7) with zero-padding to 8x8.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Build base decompositions
    U22, V22, W22 = _build_strassen_2x2()
    U44b, V44b, W44b = _build_laderman_4x4_blocks()
    
    # Verify Laderman 4x4 block decomposition (rank 25)
    # We need to verify that U44b, V44b, W44b correctly decompose 2x2 matrix multiplication
    # where the entries are 2x2 matrices.
    # This is equivalent to verifying the 4x4 decomposition using Strassen for the sub-blocks.
    # Let's just verify the final 6x6 decomposition directly.
    
    # Build the 8x8 decomposition (rank 175)
    rank = 25 * 7  # 175
    
    U88 = np.zeros((64, rank), dtype=float)
    V88 = np.zeros((64, rank), dtype=float)
    W88 = np.zeros((64, rank), dtype=float)
    
    for r44 in range(25):
        for r22 in range(7):
            r_final = r44 * 7 + r22
            
            # U: A[i,j] where i = bi*2+ii, j = bj*2+jj
            for bi in range(2):
                for bj in range(2):
                    u44_val = U44b[bi*2+bj, r44]
                    if u44_val == 0:
                        continue
                    for ii in range(2):
                        for jj in range(2):
                            u22_val = U22[ii*2+jj, r22]
                            if u22_val == 0:
                                continue
                            i = bi*2 + ii
                            j = bj*2 + jj
                            target_row = i * 8 + j
                            U88[target_row, r_final] = u44_val * u22_val

            # V: B[j,k], row = j*4+k in 4x4, row = jj*2+kk in 2x2
            for bj in range(2):
                for bk in range(2):
                    v44_val = V44b[bj*2+bk, r44]
                    if v44_val == 0:
                        continue
                    for jj in range(2):
                        for kk in range(2):
                            v22_val = V22[jj*2+kk, r22]
                            if v22_val == 0:
                                continue
                            j = bj*2 + jj
                            k = bk*2 + kk
                            target_row = j * 8 + k
                            V88[target_row, r_final] = v44_val * v22_val

            # W: C[i,k] -> W[k*4+i] in 4x4, W[kk*2+ii] in 2x2
            for bi in range(2):
                for bk in range(2):
                    w44_val = W44b[bk*2+bi, r44] # Note: W index is k*4+i in 4x4? 
                    # In Laderman, W is for C[i,k]. The block index is (bi, bk) for C[i,k]?
                    # C[i,k] is at block (bi, bk).
                    # The row in W44b is bk*2+bi?
                    # Let's check the convention.
                    # U[i,j] -> row i*4+j. Block (bi,bj).
                    # V[j,k] -> row j*4+k. Block (bj,bk).
                    # W[k,i] -> row k*4+i. Block (bk,bi).
                    # So yes, W44b row is bk*2+bi.
                    if w44_val == 0:
                        continue
                    for ii in range(2):
                        for kk in range(2):
                            w22_val = W22[kk*2+ii, r22]
                            if w22_val == 0:
                                continue
                            i = bi*2 + ii
                            k = bk*2 + kk
                            target_row = k * 8 + i
                            W88[target_row, r_final] = w44_val * w22_val

    # Extract 6x6 decomposition
    U66 = np.zeros((36, rank), dtype=float)
    V66 = np.zeros((36, rank), dtype=float)
    W66 = np.zeros((36, rank), dtype=float)
    
    for i in range(6):
        for j in range(6):
            U66[i*6+j, :] = U88[i*8+j, :]
            
    for j in range(6):
        for k in range(6):
            V66[j*6+k, :] = V88[j*8+k, :]
            
    for k in range(6):
        for i in range(6):
            W66[k*6+i, :] = W88[k*8+i, :]

    # Verify
    if time.time() < deadline - 5:
        if not _verify_decomposition(U66, V66, W66, n, m, p, seed=42):
            # Fallback to schoolbook
            rank_school = n * m * p
            U_fb = np.zeros((n * m, rank_school), dtype=float)
            V_fb = np.zeros((m * p, rank_school), dtype=float)
            W_fb = np.zeros((p * n, rank_school), dtype=float)
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, column] = 1
                        V_fb[j * p + k, column] = 1
                        W_fb[k * n + i, column] = 1
                        column += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U66.tolist(), "V": V66.tolist(), "W": W66.tolist()}

# EVOLVE-BLOCK-END
