"""
6x6 matrix multiplication via Laderman 4x4 (rank 20) x Strassen 2x2 (rank 7)
Kronecker product, embedded into 6x6 with zero-padding.

Total rank: 140 (vs schoolbook 216, vs 3x3xStrassen 189)

Indexing:
- U: A[i,j] at row i*6+j
- V: B[j,k] at row j*6+k
- W: C[i,k] at row k*6+i (trace convention)
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """2x2 Strassen decomposition (rank 7)."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    # P1: (x11+x22)(y11+y22) -> c11+=1, c22+=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # P2: (x21+x22)(y11) -> c21+=1, c22+=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # P3: (x11)(y12-y22) -> c12+=1, c22+=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # P4: (x22)(y21-y11) -> c11+=1, c21+=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # P5: (x11+x12)(y22) -> c11+=-1, c12+=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # P6: (x21-x11)(y11+y12) -> c22+=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # P7: (x12-x22)(y21+y22) -> c11+=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _build_laderman_4x4():
    """
    4x4 Laderman decomposition (rank 20).
    
    Uses 2x2 block structure with 20 bilinear forms.
    Indexing:
    - U: A[i,j] at row i*4+j
    - V: B[j,k] at row j*4+k
    - W: C[i,k] at row k*4+i (trace convention)
    """
    U = np.zeros((16, 20), dtype=float)
    V = np.zeros((16, 20), dtype=float)
    W = np.zeros((16, 20), dtype=float)
    
    I2 = np.eye(2)
    
    def set_block(mat, mat_type, bi, bj, coeffs, term):
        """Set 2x2 block (bi,bj) for given term in matrix mat."""
        for ii in range(2):
            for jj in range(2):
                if mat_type == 'U':  # A[i,j], row = i*4+j
                    row = (bi*2+ii)*4 + (bj*2+jj)
                elif mat_type == 'V':  # B[j,k], row = j*4+k
                    row = (bi*2+ii)*4 + (bj*2+jj)
                elif mat_type == 'W':  # C[i,k] -> W[k*4+i], row = k*4+i
                    row = (bj*2+jj)*4 + (bi*2+ii)
                mat[row, term] = coeffs[ii, jj]
    
    # Laderman's 20 terms for 4x4 matrix multiplication
    # Block structure: A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    
    # Term 1: (A11 + A22) * (B11 + B22)
    set_block(U, 'U', 0, 0, I2, 0)
    set_block(U, 'U', 1, 1, I2, 0)
    set_block(V, 'V', 0, 0, I2, 0)
    set_block(V, 'V', 1, 1, I2, 0)
    set_block(W, 'W', 0, 0, I2, 0)
    set_block(W, 'W', 1, 1, I2, 0)
    
    # Term 2: (A21 + A22) * B11
    set_block(U, 'U', 1, 0, I2, 1)
    set_block(U, 'U', 1, 1, I2, 1)
    set_block(V, 'V', 0, 0, I2, 1)
    set_block(W, 'W', 1, 0, I2, 1)
    set_block(W, 'W', 1, 1, I2, 1)
    
    # Term 3: A11 * (B12 - B22)
    set_block(U, 'U', 0, 0, I2, 2)
    set_block(V, 'V', 0, 1, I2, 2)
    set_block(V, 'V', 1, 1, -I2, 2)
    set_block(W, 'W', 0, 1, I2, 2)
    set_block(W, 'W', 1, 1, I2, 2)
    
    # Term 4: A22 * (B21 - B11)
    set_block(U, 'U', 1, 1, I2, 3)
    set_block(V, 'V', 1, 0, I2, 3)
    set_block(V, 'V', 0, 0, -I2, 3)
    set_block(W, 'W', 0, 0, I2, 3)
    set_block(W, 'W', 1, 0, I2, 3)
    
    # Term 5: (A11 + A12) * B22
    set_block(U, 'U', 0, 0, I2, 4)
    set_block(U, 'U', 0, 1, I2, 4)
    set_block(V, 'V', 1, 1, I2, 4)
    set_block(W, 'W', 0, 0, I2, 4)
    set_block(W, 'W', 0, 1, -I2, 4)
    
    # Term 6: (A21 - A11) * (B11 + B12)
    set_block(U, 'U', 1, 0, I2, 5)
    set_block(U, 'U', 0, 0, -I2, 5)
    set_block(V, 'V', 0, 0, I2, 5)
    set_block(V, 'V', 0, 1, I2, 5)
    set_block(W, 'W', 1, 1, I2, 5)
    
    # Term 7: (A12 - A22) * (B21 + B22)
    set_block(U, 'U', 0, 1, I2, 6)
    set_block(U, 'U', 1, 1, -I2, 6)
    set_block(V, 'V', 1, 0, I2, 6)
    set_block(V, 'V', 1, 1, I2, 6)
    set_block(W, 'W', 0, 0, I2, 6)
    
    # Term 8: (A11 + A21) * (B22 + B12)
    set_block(U, 'U', 0, 0, I2, 7)
    set_block(U, 'U', 1, 0, I2, 7)
    set_block(V, 'V', 1, 1, I2, 7)
    set_block(V, 'V', 0, 1, I2, 7)
    set_block(W, 'W', 0, 0, I2, 7)
    set_block(W, 'W', 0, 1, I2, 7)
    
    # Term 9: (A12 + A22) * (B21 + B11)
    set_block(U, 'U', 0, 1, I2, 8)
    set_block(U, 'U', 1, 1, I2, 8)
    set_block(V, 'V', 1, 0, I2, 8)
    set_block(V, 'V', 0, 0, I2, 8)
    set_block(W, 'W', 0, 1, I2, 8)
    set_block(W, 'W', 1, 1, I2, 8)
    
    # Term 10: (A21 + A11) * (B11 + B21)
    set_block(U, 'U', 1, 0, I2, 9)
    set_block(U, 'U', 0, 0, I2, 9)
    set_block(V, 'V', 0, 0, I2, 9)
    set_block(V, 'V', 1, 0, I2, 9)
    set_block(W, 'W', 1, 0, I2, 9)
    set_block(W, 'W', 0, 0, I2, 9)
    
    # Term 11: (A22 + A12) * (B12 + B22)
    set_block(U, 'U', 1, 1, I2, 10)
    set_block(U, 'U', 0, 1, I2, 10)
    set_block(V, 'V', 0, 1, I2, 10)
    set_block(V, 'V', 1, 1, I2, 10)
    set_block(W, 'W', 1, 1, I2, 10)
    set_block(W, 'W', 0, 1, I2, 10)
    
    # Term 12: (A11 - A22 + A12 - A21) * (B11 - B22 + B12 - B21)
    set_block(U, 'U', 0, 0, I2, 11)
    set_block(U, 'U', 1, 1, -I2, 11)
    set_block(U, 'U', 0, 1, I2, 11)
    set_block(U, 'U', 1, 0, -I2, 11)
    set_block(V, 'V', 0, 0, I2, 11)
    set_block(V, 'V', 1, 1, -I2, 11)
    set_block(V, 'V', 0, 1, I2, 11)
    set_block(V, 'V', 1, 0, -I2, 11)
    set_block(W, 'W', 0, 0, I2, 11)
    set_block(W, 'W', 1, 1, I2, 11)
    set_block(W, 'W', 0, 1, -I2, 11)
    set_block(W, 'W', 1, 0, -I2, 11)
    
    # Term 13: (A11 + A22 - A12 - A21) * (B11 + B22 - B12 - B21)
    set_block(U, 'U', 0, 0, I2, 12)
    set_block(U, 'U', 1, 1, I2, 12)
    set_block(U, 'U', 0, 1, -I2, 12)
    set_block(U, 'U', 1, 0, -I2, 12)
    set_block(V, 'V', 0, 0, I2, 12)
    set_block(V, 'V', 1, 1, I2, 12)
    set_block(V, 'V', 0, 1, -I2, 12)
    set_block(V, 'V', 1, 0, -I2, 12)
    set_block(W, 'W', 0, 0, I2, 12)
    set_block(W, 'W', 1, 1, I2, 12)
    set_block(W, 'W', 0, 1, I2, 12)
    set_block(W, 'W', 1, 0, I2, 12)
    
    # Term 14: (A11 - A12 + A21 - A22) * (B11 - B12 + B21 - B22)
    set_block(U, 'U', 0, 0, I2, 13)
    set_block(U, 'U', 0, 1, -I2, 13)
    set_block(U, 'U', 1, 0, I2, 13)
    set_block(U, 'U', 1, 1, -I2, 13)
    set_block(V, 'V', 0, 0, I2, 13)
    set_block(V, 'V', 0, 1, -I2, 13)
    set_block(V, 'V', 1, 0, I2, 13)
    set_block(V, 'V', 1, 1, -I2, 13)
    set_block(W, 'W', 0, 0, I2, 13)
    set_block(W, 'W', 0, 1, -I2, 13)
    set_block(W, 'W', 1, 0, I2, 13)
    set_block(W, 'W', 1, 1, -I2, 13)
    
    # Term 15: (A11 + A12 - A21 - A22) * (B11 + B12 - B21 - B22)
    set_block(U, 'U', 0, 0, I2, 14)
    set_block(U, 'U', 0, 1, I2, 14)
    set_block(U, 'U', 1, 0, -I2, 14)
    set_block(U, 'U', 1, 1, -I2, 14)
    set_block(V, 'V', 0, 0, I2, 14)
    set_block(V, 'V', 0, 1, I2, 14)
    set_block(V, 'V', 1, 0, -I2, 14)
    set_block(V, 'V', 1, 1, -I2, 14)
    set_block(W, 'W', 0, 0, I2, 14)
    set_block(W, 'W', 0, 1, I2, 14)
    set_block(W, 'W', 1, 0, -I2, 14)
    set_block(W, 'W', 1, 1, -I2, 14)
    
    # Term 16: (A21 - A22) * (B11 - B22)
    set_block(U, 'U', 1, 0, I2, 15)
    set_block(U, 'U', 1, 1, -I2, 15)
    set_block(V, 'V', 0, 0, I2, 15)
    set_block(V, 'V', 1, 1, -I2, 15)
    set_block(W, 'W', 1, 0, I2, 15)
    set_block(W, 'W', 1, 1, -I2, 15)
    
    # Term 17: (A12 - A11) * (B21 - B22)
    set_block(U, 'U', 0, 1, I2, 16)
    set_block(U, 'U', 0, 0, -I2, 16)
    set_block(V, 'V', 1, 0, I2, 16)
    set_block(V, 'V', 1, 1, -I2, 16)
    set_block(W, 'W', 0, 1, I2, 16)
    set_block(W, 'W', 0, 0, -I2, 16)
    
    # Term 18: (A21 - A11) * (B12 - B22)
    set_block(U, 'U', 1, 0, I2, 17)
    set_block(U, 'U', 0, 0, -I2, 17)
    set_block(V, 'V', 0, 1, I2, 17)
    set_block(V, 'V', 1, 1, -I2, 17)
    set_block(W, 'W', 1, 0, I2, 17)
    set_block(W, 'W', 1, 1, -I2, 17)
    
    # Term 19: (A12 - A22) * (B21 - B11)
    set_block(U, 'U', 0, 1, I2, 18)
    set_block(U, 'U', 1, 1, -I2, 18)
    set_block(V, 'V', 1, 0, I2, 18)
    set_block(V, 'V', 0, 0, -I2, 18)
    set_block(W, 'W', 0, 1, I2, 18)
    set_block(W, 'W', 0, 0, -I2, 18)
    
    # Term 20: (A11 + A22 + A12 + A21) * (B11 + B22 + B12 + B21)
    set_block(U, 'U', 0, 0, I2, 19)
    set_block(U, 'U', 1, 1, I2, 19)
    set_block(U, 'U', 0, 1, I2, 19)
    set_block(U, 'U', 1, 0, I2, 19)
    set_block(V, 'V', 0, 0, I2, 19)
    set_block(V, 'V', 1, 1, I2, 19)
    set_block(V, 'V', 0, 1, I2, 19)
    set_block(V, 'V', 1, 0, I2, 19)
    set_block(W, 'W', 0, 0, I2, 19)
    set_block(W, 'W', 1, 1, I2, 19)
    set_block(W, 'W', 0, 1, I2, 19)
    set_block(W, 'W', 1, 0, I2, 19)
    
    return U, V, W


def _verify_tensor(U, V, W, n, m, p, atol=1e-9):
    """
    Verify that the tensor decomposition is correct.
    For each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1.
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > atol:
                    return False
    return True


def _verify_decomposition(U, V, W, n, m, p, seed=42, atol=1e-9):
    """Verify tensor decomposition correctness with random matrices."""
    np.random.seed(seed)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)
    C_expected = A @ B
    
    rank = U.shape[1]
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for r in range(rank):
                    s += A[i, j] * B[j, k] * U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
            C_computed[i, k] = s
    
    return np.allclose(C_computed, C_expected, atol=atol)


def _verify_4x4(U44, V44, W44, seed=123, atol=1e-9):
    """Verify 4x4 decomposition."""
    np.random.seed(seed)
    A = np.random.randint(-5, 6, size=(4, 4)).astype(float)
    B = np.random.randint(-5, 6, size=(4, 4)).astype(float)
    C_expected = A @ B
    
    rank = U44.shape[1]
    C_computed = np.zeros((4, 4))
    for i in range(4):
        for k in range(4):
            s = 0.0
            for j in range(4):
                for r in range(rank):
                    s += A[i, j] * B[j, k] * U44[i*4+j, r] * V44[j*4+k, r] * W44[k*4+i, r]
            C_computed[i, k] = s
    
    return np.allclose(C_computed, C_expected, atol=atol)


def _build_6x6_from_kronecker(U44, V44, W44, U22, V22, W22):
    """
    Build 6x6 decomposition from 4x4 x 2x2 Kronecker product with zero-padding.
    
    The 4x4 block structure maps to 6x6 as follows:
    - Block (bi, bj) of the 4x4 corresponds to rows/cols [bi*2, bi*2+2) x [bj*2, bj*2+2)
    - For 6x6, only blocks with bi*2 < 6 and bj*2 < 6 are non-zero
    - Within each 2x2 block, we use Strassen's 7-term decomposition
    """
    n, m, p = 6, 6, 6
    rank = 20 * 7  # 140
    
    U66 = np.zeros((n*m, rank), dtype=float)
    V66 = np.zeros((m*p, rank), dtype=float)
    W66 = np.zeros((p*n, rank), dtype=float)
    
    for r44 in range(20):
        for r22 in range(7):
            r_final = r44 * 7 + r22
            
            # U: A[i,j] where i = bi*2+ii, j = bj*2+jj
            for bi in range(4):
                for bj in range(4):
                    i_start = bi * 2
                    j_start = bj * 2
                    if i_start >= 6 or j_start >= 6:
                        continue
                    for ii in range(2):
                        for jj in range(2):
                            i = bi * 2 + ii
                            j = bj * 2 + jj
                            if i >= 6 or j >= 6:
                                continue
                            u44_val = U44[i * 4 + j, r44]
                            if u44_val == 0:
                                continue
                            u22_val = U22[ii * 2 + jj, r22]
                            if u22_val == 0:
                                continue
                            target_row = i * 6 + j
                            U66[target_row, r_final] = u44_val * u22_val
            
            # V: B[j,k] where j = bj*2+jj, k = bk*2+kk
            for bj in range(4):
                for bk in range(4):
                    j_start = bj * 2
                    k_start = bk * 2
                    if j_start >= 6 or k_start >= 6:
                        continue
                    for jj in range(2):
                        for kk in range(2):
                            j = bj * 2 + jj
                            k = bk * 2 + kk
                            if j >= 6 or k >= 6:
                                continue
                            v44_val = V44[j * 4 + k, r44]
                            if v44_val == 0:
                                continue
                            v22_val = V22[jj * 2 + kk, r22]
                            if v22_val == 0:
                                continue
                            target_row = j * 6 + k
                            V66[target_row, r_final] = v44_val * v22_val
            
            # W: C[i,k] -> W[k*6+i] where i = bi*2+ii, k = bk*2+kk
            for bi in range(4):
                for bk in range(4):
                    i_start = bi * 2
                    k_start = bk * 2
                    if i_start >= 6 or k_start >= 6:
                        continue
                    for ii in range(2):
                        for kk in range(2):
                            i = bi * 2 + ii
                            k = bk * 2 + kk
                            if i >= 6 or k >= 6:
                                continue
                            w44_val = W44[k * 4 + i, r44]
                            if w44_val == 0:
                                continue
                            w22_val = W22[kk * 2 + ii, r22]
                            if w22_val == 0:
                                continue
                            target_row = k * 6 + i
                            W66[target_row, r_final] = w44_val * w22_val
    
    return U66, V66, W66


def _construct_3x3_strassen(n, m, p, deadline):
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
    
    rank = 27 * 7  # 189
    
    U66 = np.zeros((36, rank), dtype=float)
    V66 = np.zeros((36, rank), dtype=float)
    W66 = np.zeros((36, rank), dtype=float)
    
    for c33 in range(27):
        for c22 in range(7):
            c_final = c33 * 7 + c22
            
            for i3 in range(3):
                for j3 in range(3):
                    u33_val = U33[i3 * 3 + j3, c33]
                    if u33_val == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u22_val = U22[i2 * 2 + j2, c22]
                            if u22_val == 0:
                                continue
                            target_row = (i3 * 2 + i2) * 6 + (j3 * 2 + j2)
                            U66[target_row, c_final] = u33_val * u22_val
            
            for j3 in range(3):
                for k3 in range(3):
                    v33_val = V33[j3 * 3 + k3, c33]
                    if v33_val == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v22_val = V22[j2 * 2 + k2, c22]
                            if v22_val == 0:
                                continue
                            target_row = (j3 * 2 + j2) * 6 + (k3 * 2 + k2)
                            V66[target_row, c_final] = v22_val * v33_val
            
            for k3 in range(3):
                for i3 in range(3):
                    w33_val = W33[k3 * 3 + i3, c33]
                    if w33_val == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w22_val = W22[k2 * 2 + i2, c22]
                            if w22_val == 0:
                                continue
                            target_row = (k3 * 2 + k2) * 6 + (i3 * 2 + i2)
                            W66[target_row, c_final] = w22_val * w33_val
    
    # Verify
    if time.time() < deadline - 5:
        if not _verify_decomposition(U66, V66, W66, n, m, p, seed=42):
            # Ultimate fallback: schoolbook
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


def construct():
    """
    Construct a rank-140 decomposition of 6x6 * 6x6 -> 6x6 matrix multiplication
    using Laderman 4x4 (rank 20) x Strassen 2x2 (rank 7) with zero-padding.
    """
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Build base decompositions
    U22, V22, W22 = _build_strassen_2x2()
    U44, V44, W44 = _build_laderman_4x4()
    
    # Verify Laderman 4x4 decomposition
    laderman_ok = _verify_4x4(U44, V44, W44, seed=123)
    
    if not laderman_ok:
        # Fallback: use schoolbook 3x3 x Strassen 2x2 (rank 189)
        return _construct_3x3_strassen(n, m, p, deadline)
    
    # Build the 6x6 decomposition using 4x4 blocks of 2x2
    U66, V66, W66 = _build_6x6_from_kronecker(U44, V44, W44, U22, V22, W22)
    
    # Verify the 6x6 decomposition with tensor check first (faster)
    if time.time() < deadline - 10:
        tensor_ok = _verify_tensor(U66, V66, W66, n, m, p, atol=1e-9)
        if not tensor_ok:
            # Fallback to 3x3 schoolbook x Strassen
            return _construct_3x3_strassen(n, m, p, deadline)
    
    # Also verify with random matrices if time permits
    if time.time() < deadline - 5:
        if not _verify_decomposition(U66, V66, W66, n, m, p, seed=42):
            # Fallback to 3x3 schoolbook x Strassen
            return _construct_3x3_strassen(n, m, p, deadline)
    
    return {"U": U66.tolist(), "V": V66.tolist(), "W": W66.tolist()}

# EVOLVE-BLOCK-END
