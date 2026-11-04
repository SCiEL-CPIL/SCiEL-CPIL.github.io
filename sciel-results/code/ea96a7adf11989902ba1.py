import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    Build the 2x2 Strassen decomposition (rank 7).
    Returns U, V, W of shape (4, 7).
    Indexing:
    U[i*2+j, r] corresponds to A[i,j]
    V[i*2+j, r] corresponds to B[i,j]
    W[i*2+j, r] corresponds to C[i,j]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    # P1: (a+d)(e+h)
    # A: a=00, d=11. B: e=00, h=11.
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C: c11+=1, c22+=1. Indices: c11=0, c22=3.
    W[0, 0] = 1; W[3, 0] = 1

    # P2: (c+d)e
    # A: c=10, d=11. B: e=00.
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C: c21+=1, c22+=-1. Indices: c21=2, c22=3.
    W[2, 1] = 1; W[3, 1] = -1

    # P3: a(f-h)
    # A: a=00. B: f=01, h=11.
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # C: c12+=1, c22+=1. Indices: c12=1, c22=3.
    W[1, 2] = 1; W[3, 2] = 1

    # P4: d(g-c)
    # A: d=11. B: g=10, c=00.
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # C: c11+=1, c21+=1. Indices: c11=0, c21=2.
    W[0, 3] = 1; W[2, 3] = 1

    # P5: (a+b)h
    # A: a=00, b=01. B: h=11.
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C: c11+=-1, c12+=1. Indices: c11=0, c12=1.
    W[0, 4] = -1; W[1, 4] = 1

    # P6: (c-a)(e+f)
    # A: c=10, a=00. B: e=00, f=01.
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C: c22+=1. Index: c22=3.
    W[3, 5] = 1

    # P7: (b-d)(g+h)
    # A: b=01, d=11. B: g=10, h=11.
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C: c11+=1. Index: c11=0.
    W[0, 6] = 1

    return U, V, W

def _build_laderman_4x4():
    """
    Build the 4x4 Laderman decomposition (rank 20).
    Returns U, V, W of shape (16, 20).
    Indexing:
    U[i*4+j, r] corresponds to A[i,j]
    V[i*4+j, r] corresponds to B[i,j]
    W[i*4+j, r] corresponds to C[i,j]
    """
    U = np.zeros((16, 20), dtype=float)
    V = np.zeros((16, 20), dtype=float)
    W = np.zeros((16, 20), dtype=float)
    
    # Helper to set a 2x2 block in the matrix
    def set_block(mat, bi, bj, coeffs, term):
        for ii in range(2):
            for jj in range(2):
                mat[(bi*2+ii)*4 + (bj*2+jj), term] = coeffs[ii, jj]

    I2 = np.eye(2)
    
    # Laderman's 20 terms
    # Term 1: (A11 + A22) * (B11 + B22)
    set_block(U, 0, 0, I2, 0)
    set_block(U, 1, 1, I2, 0)
    set_block(V, 0, 0, I2, 0)
    set_block(V, 1, 1, I2, 0)
    set_block(W, 0, 0, I2, 0)
    set_block(W, 1, 1, I2, 0)
    
    # Term 2: (A21 + A22) * B11
    set_block(U, 1, 0, I2, 1)
    set_block(U, 1, 1, I2, 1)
    set_block(V, 0, 0, I2, 1)
    set_block(W, 1, 0, I2, 1)
    set_block(W, 1, 1, I2, 1)
    
    # Term 3: A11 * (B12 - B22)
    set_block(U, 0, 0, I2, 2)
    set_block(V, 0, 1, I2, 2)
    set_block(V, 1, 1, -I2, 2)
    set_block(W, 0, 1, I2, 2)
    set_block(W, 1, 1, I2, 2)
    
    # Term 4: A22 * (B21 - B11)
    set_block(U, 1, 1, I2, 3)
    set_block(V, 1, 0, I2, 3)
    set_block(V, 0, 0, -I2, 3)
    set_block(W, 0, 0, I2, 3)
    set_block(W, 1, 0, I2, 3)
    
    # Term 5: (A11 + A12) * B22
    set_block(U, 0, 0, I2, 4)
    set_block(U, 0, 1, I2, 4)
    set_block(V, 1, 1, I2, 4)
    set_block(W, 0, 0, I2, 4)
    set_block(W, 0, 1, -I2, 4)
    
    # Term 6: (A21 - A11) * (B11 + B12)
    set_block(U, 1, 0, I2, 5)
    set_block(U, 0, 0, -I2, 5)
    set_block(V, 0, 0, I2, 5)
    set_block(V, 0, 1, I2, 5)
    set_block(W, 1, 1, I2, 5)
    
    # Term 7: (A12 - A22) * (B21 + B22)
    set_block(U, 0, 1, I2, 6)
    set_block(U, 1, 1, -I2, 6)
    set_block(V, 1, 0, I2, 6)
    set_block(V, 1, 1, I2, 6)
    set_block(W, 0, 0, I2, 6)
    
    # Term 8: (A11 + A21) * (B22 + B12)
    set_block(U, 0, 0, I2, 7)
    set_block(U, 1, 0, I2, 7)
    set_block(V, 1, 1, I2, 7)
    set_block(V, 0, 1, I2, 7)
    set_block(W, 0, 0, I2, 7)
    set_block(W, 0, 1, I2, 7)
    
    # Term 9: (A12 + A22) * (B21 + B11)
    set_block(U, 0, 1, I2, 8)
    set_block(U, 1, 1, I2, 8)
    set_block(V, 1, 0, I2, 8)
    set_block(V, 0, 0, I2, 8)
    set_block(W, 0, 1, I2, 8)
    set_block(W, 1, 1, I2, 8)
    
    # Term 10: (A21 + A11) * (B11 + B21)
    set_block(U, 1, 0, I2, 9)
    set_block(U, 0, 0, I2, 9)
    set_block(V, 0, 0, I2, 9)
    set_block(V, 1, 0, I2, 9)
    set_block(W, 1, 0, I2, 9)
    set_block(W, 0, 0, I2, 9)
    
    # Term 11: (A22 + A12) * (B12 + B22)
    set_block(U, 1, 1, I2, 10)
    set_block(U, 0, 1, I2, 10)
    set_block(V, 0, 1, I2, 10)
    set_block(V, 1, 1, I2, 10)
    set_block(W, 1, 1, I2, 10)
    set_block(W, 0, 1, I2, 10)
    
    # Term 12: (A11 - A22 + A12 - A21) * (B11 - B22 + B12 - B21)
    set_block(U, 0, 0, I2, 11)
    set_block(U, 1, 1, -I2, 11)
    set_block(U, 0, 1, I2, 11)
    set_block(U, 1, 0, -I2, 11)
    set_block(V, 0, 0, I2, 11)
    set_block(V, 1, 1, -I2, 11)
    set_block(V, 0, 1, I2, 11)
    set_block(V, 1, 0, -I2, 11)
    set_block(W, 0, 0, I2, 11)
    set_block(W, 1, 1, I2, 11)
    set_block(W, 0, 1, -I2, 11)
    set_block(W, 1, 0, -I2, 11)
    
    # Term 13: (A11 + A22 - A12 - A21) * (B11 + B22 - B12 - B21)
    set_block(U, 0, 0, I2, 12)
    set_block(U, 1, 1, I2, 12)
    set_block(U, 0, 1, -I2, 12)
    set_block(U, 1, 0, -I2, 12)
    set_block(V, 0, 0, I2, 12)
    set_block(V, 1, 1, I2, 12)
    set_block(V, 0, 1, -I2, 12)
    set_block(V, 1, 0, -I2, 12)
    set_block(W, 0, 0, I2, 12)
    set_block(W, 1, 1, I2, 12)
    set_block(W, 0, 1, I2, 12)
    set_block(W, 1, 0, I2, 12)
    
    # Term 14: (A11 - A12 + A21 - A22) * (B11 - B12 + B21 - B22)
    set_block(U, 0, 0, I2, 13)
    set_block(U, 0, 1, -I2, 13)
    set_block(U, 1, 0, I2, 13)
    set_block(U, 1, 1, -I2, 13)
    set_block(V, 0, 0, I2, 13)
    set_block(V, 0, 1, -I2, 13)
    set_block(V, 1, 0, I2, 13)
    set_block(V, 1, 1, -I2, 13)
    set_block(W, 0, 0, I2, 13)
    set_block(W, 0, 1, -I2, 13)
    set_block(W, 1, 0, I2, 13)
    set_block(W, 1, 1, -I2, 13)
    
    # Term 15: (A11 + A12 - A21 - A22) * (B11 + B12 - B21 - B22)
    set_block(U, 0, 0, I2, 14)
    set_block(U, 0, 1, I2, 14)
    set_block(U, 1, 0, -I2, 14)
    set_block(U, 1, 1, -I2, 14)
    set_block(V, 0, 0, I2, 14)
    set_block(V, 0, 1, I2, 14)
    set_block(V, 1, 0, -I2, 14)
    set_block(V, 1, 1, -I2, 14)
    set_block(W, 0, 0, I2, 14)
    set_block(W, 0, 1, I2, 14)
    set_block(W, 1, 0, -I2, 14)
    set_block(W, 1, 1, -I2, 14)
    
    # Term 16: (A21 - A22) * (B11 - B22)
    set_block(U, 1, 0, I2, 15)
    set_block(U, 1, 1, -I2, 15)
    set_block(V, 0, 0, I2, 15)
    set_block(V, 1, 1, -I2, 15)
    set_block(W, 1, 0, I2, 15)
    set_block(W, 1, 1, -I2, 15)
    
    # Term 17: (A12 - A11) * (B21 - B22)
    set_block(U, 0, 1, I2, 16)
    set_block(U, 0, 0, -I2, 16)
    set_block(V, 1, 0, I2, 16)
    set_block(V, 1, 1, -I2, 16)
    set_block(W, 0, 1, I2, 16)
    set_block(W, 0, 0, -I2, 16)
    
    # Term 18: (A21 - A11) * (B12 - B22)
    set_block(U, 1, 0, I2, 17)
    set_block(U, 0, 0, -I2, 17)
    set_block(V, 0, 1, I2, 17)
    set_block(V, 1, 1, -I2, 17)
    set_block(W, 1, 0, I2, 17)
    set_block(W, 1, 1, -I2, 17)
    
    # Term 19: (A12 - A22) * (B21 - B11)
    set_block(U, 0, 1, I2, 18)
    set_block(U, 1, 1, -I2, 18)
    set_block(V, 1, 0, I2, 18)
    set_block(V, 0, 0, -I2, 18)
    set_block(W, 0, 1, I2, 18)
    set_block(W, 0, 0, -I2, 18)
    
    # Term 20: (A11 + A22 + A12 + A21) * (B11 + B22 + B12 + B21)
    set_block(U, 0, 0, I2, 19)
    set_block(U, 1, 1, I2, 19)
    set_block(U, 0, 1, I2, 19)
    set_block(U, 1, 0, I2, 19)
    set_block(V, 0, 0, I2, 19)
    set_block(V, 1, 1, I2, 19)
    set_block(V, 0, 1, I2, 19)
    set_block(V, 1, 0, I2, 19)
    set_block(W, 0, 0, I2, 19)
    set_block(W, 1, 1, I2, 19)
    set_block(W, 0, 1, I2, 19)
    set_block(W, 1, 0, I2, 19)
    
    return U, V, W

def _verify_decomposition(U, V, W, n):
    """Verify that U, V, W form a valid decomposition for n x n matrix multiplication."""
    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, n)).astype(float)
    B = np.random.randint(-5, 6, size=(n, n)).astype(float)
    C_expected = A @ B
    
    C_computed = np.zeros((n, n))
    rank = U.shape[1]
    for i in range(n):
        for k in range(n):
            s = 0.0
            for j in range(n):
                for r in range(rank):
                    s += A[i, j] * B[j, k] * U[i*n + j, r] * V[j*n + k, r] * W[k*n + i, r]
            C_computed[i, k] = s
            
    return np.allclose(C_computed, C_expected, atol=1e-9)

def _construct_6x6_laderman_strassen():
    """
    Construct a rank-140 decomposition for 6x6 matrix multiplication
    by embedding it into an 8x8 Laderman-Strassen decomposition.
    """
    U22, V22, W22 = _build_strassen_2x2()
    U44, V44, W44 = _build_laderman_4x4()
    
    # Verify Laderman 4x4
    if not _verify_decomposition(U44, V44, W44, 4):
        return None
        
    # Construct 8x8 factors
    rank_8x8 = 20 * 7
    U8 = np.zeros((64, rank_8x8), dtype=float)
    V8 = np.zeros((64, rank_8x8), dtype=float)
    W8 = np.zeros((64, rank_8x8), dtype=float)
    
    for r44 in range(20):
        for r22 in range(7):
            r_final = r44 * 7 + r22
            
            # U: A[i,j] in 8x8 -> i = 2*bi+ii, j = 2*bj+jj
            for bi in range(4):
                for bj in range(4):
                    for ii in range(2):
                        for jj in range(2):
                            i = 2*bi + ii
                            j = 2*bj + jj
                            u44_val = U44[bi*4 + bj, r44]
                            u22_val = U22[ii*2 + jj, r22]
                            if u44_val != 0 and u22_val != 0:
                                U8[i*8 + j, r_final] = u44_val * u22_val
                                
            # V: B[j,k] in 8x8 -> j = 2*bj+jj, k = 2*bk+kk
            for bj in range(4):
                for bk in range(4):
                    for jj in range(2):
                        for kk in range(2):
                            j = 2*bj + jj
                            k = 2*bk + kk
                            v44_val = V44[bj*4 + bk, r44]
                            v22_val = V22[jj*2 + kk, r22]
                            if v44_val != 0 and v22_val != 0:
                                V8[j*8 + k, r_final] = v44_val * v22_val
                                
            # W: C[i,k] in 8x8 -> i = 2*bi+ii, k = 2*bk+kk
            # W8 index is i*8 + k
            for bi in range(4):
                for bk in range(4):
                    for ii in range(2):
                        for kk in range(2):
                            i = 2*bi + ii
                            k = 2*bk + kk
                            w44_val = W44[bi*4 + bk, r44]
                            w22_val = W22[ii*2 + kk, r22]
                            if w44_val != 0 and w22_val != 0:
                                W8[i*8 + k, r_final] = w44_val * w22_val
                                
    # Verify 8x8
    if not _verify_decomposition(U8, V8, W8, 8):
        return None
        
    # Extract 6x6
    U6 = np.zeros((36, rank_8x8), dtype=float)
    V6 = np.zeros((36, rank_8x8), dtype=float)
    W6 = np.zeros((36, rank_8x8), dtype=float)
    
    for i in range(6):
        for j in range(6):
            U6[i*6 + j, :] = U8[i*8 + j, :]
            V6[j*6 + i, :] = V8[j*8 + i, :] # Note: V index is j*6+k, here k=i? No.
            # V[j,k] in 6x6 corresponds to V8[j*8+k].
            # We need to fill V6[j*6+k, r] = V8[j*8+k, r]
            
    # Correctly fill V6 and W6
    for j in range(6):
        for k in range(6):
            V6[j*6 + k, :] = V8[j*8 + k, :]
            
    for i in range(6):
        for k in range(6):
            # W6 index is k*6 + i
            # W8 index is i*8 + k
            W6[k*6 + i, :] = W8[i*8 + k, :]
            
    # Verify 6x6
    if not _verify_decomposition(U6, V6, W6, 6):
        return None
        
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}

def _construct_3x3_strassen():
    """Fallback: 3x3 schoolbook x 2x2 Strassen, rank 189."""
    U22, V22, W22 = _build_strassen_2x2()
    
    U33 = np.zeros((9, 27), dtype=float)
    V33 = np.zeros((9, 27), dtype=float)
    W33 = np.zeros((9, 27), dtype=float)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U33[i * 3 + j, col] = 1
                V33[j * 3 + k, col] = 1
                W33[k * 3 + i, col] = 1
                col += 1
                
    if not _verify_decomposition(U33, V33, W33, 3):
        return None
        
    rank = 27 * 7
    U6 = np.zeros((36, rank), dtype=float)
    V6 = np.zeros((36, rank), dtype=float)
    W6 = np.zeros((36, rank), dtype=float)
    
    for c33 in range(27):
        for c22 in range(7):
            c_final = c33 * 7 + c22
            for i3 in range(3):
                for j3 in range(3):
                    u33_val = U33[i3 * 3 + j3, c33]
                    if u33_val == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u22_val = U22[i2 * 2 + j2, c22]
                            if u22_val == 0: continue
                            target_row = (i3 * 2 + i2) * 6 + (j3 * 2 + j2)
                            U6[target_row, c_final] = u33_val * u22_val
                            
            for j3 in range(3):
                for k3 in range(3):
                    v33_val = V33[j3 * 3 + k3, c33]
                    if v33_val == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v22_val = V22[j2 * 2 + k2, c22]
                            if v22_val == 0: continue
                            target_row = (j3 * 2 + j2) * 6 + (k3 * 2 + k2)
                            V6[target_row, c_final] = v33_val * v22_val
                            
            for k3 in range(3):
                for i3 in range(3):
                    w33_val = W33[k3 * 3 + i3, c33]
                    if w33_val == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w22_val = W22[k2 * 2 + i2, c22]
                            if w22_val == 0: continue
                            target_row = (k3 * 2 + k2) * 6 + (i3 * 2 + i2)
                            W6[target_row, c_final] = w33_val * w22_val
                            
    if not _verify_decomposition(U6, V6, W6, 6):
        return None
        
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    result = _construct_6x6_laderman_strassen()
    if result:
        return result
        
    if time.time() < deadline - 5:
        result = _construct_3x3_strassen()
        if result:
            return result
            
    # Ultimate fallback: Schoolbook
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
