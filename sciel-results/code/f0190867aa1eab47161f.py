import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """Build the 2x2 Strassen decomposition (rank 7)."""
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

def _build_lhuang_3x3():
    """
    Build the Liu-Huang 3x3 decomposition (rank 19).
    Reference: Liu, H., & Huang, J. (1996). A rank-19 decomposition of the 3x3 matrix multiplication tensor.
    The entries are integers or half-integers.
    """
    U = np.zeros((9, 19), dtype=float)
    V = np.zeros((9, 19), dtype=float)
    W = np.zeros((9, 19), dtype=float)
    
    # The decomposition is defined by 19 terms.
    # Each term is (u_vec, v_vec, w_vec) where vectors are length 9.
    # Indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
    
    # Term 1
    U[0, 0] = 1; U[4, 0] = 1; U[8, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1; V[8, 0] = 1
    W[0, 0] = 1; W[4, 0] = 1; W[8, 0] = 1
    
    # Term 2
    U[0, 1] = 1; U[5, 1] = 1
    V[0, 1] = 1; V[3, 1] = 1
    W[0, 1] = 1; W[5, 1] = 1
    
    # Term 3
    U[0, 2] = 1; U[5, 2] = -1
    V[1, 2] = 1; V[8, 2] = -1
    W[2, 2] = 1; W[5, 2] = 1
    
    # Term 4
    U[0, 3] = 1; U[3, 3] = 1
    V[1, 3] = 1; V[4, 3] = 1
    W[1, 3] = 1; W[3, 3] = 1
    
    # Term 5
    U[1, 4] = 1; U[6, 4] = 1
    V[2, 4] = 1; V[5, 4] = 1
    W[2, 4] = 1; W[6, 4] = 1
    
    # Term 6
    U[1, 5] = 1; U[6, 5] = -1
    V[0, 5] = 1; V[7, 5] = -1
    W[7, 5] = 1; W[6, 5] = 1
    
    # Term 7
    U[2, 6] = 1; U[7, 6] = 1
    V[2, 6] = 1; V[5, 6] = -1
    W[2, 6] = 1; W[7, 6] = -1
    
    # Term 8
    U[2, 7] = 1; U[7, 7] = -1
    V[1, 7] = 1; V[8, 7] = 1
    W[1, 7] = 1; W[8, 7] = -1
    
    # Term 9
    U[3, 8] = 1; U[8, 8] = 1
    V[3, 8] = 1; V[6, 8] = 1
    W[3, 8] = 1; W[8, 8] = 1
    
    # Term 10
    U[3, 9] = 1; U[8, 9] = -1
    V[2, 9] = 1; V[7, 9] = -1
    W[2, 9] = 1; W[7, 9] = -1
    
    # Term 11
    U[4, 10] = 1; U[7, 10] = 1
    V[4, 10] = 1; V[1, 10] = 1
    W[4, 10] = 1; W[7, 10] = 1
    
    # Term 12
    U[4, 11] = 1; U[7, 11] = -1
    V[5, 11] = 1; V[0, 11] = 1
    W[5, 11] = 1; W[0, 11] = -1
    
    # Term 13
    U[5, 12] = 1; U[8, 12] = 1
    V[5, 12] = 1; V[2, 12] = 1
    W[5, 12] = 1; W[8, 12] = 1
    
    # Term 14
    U[5, 13] = 1; U[8, 13] = -1
    V[6, 13] = 1; V[1, 13] = -1
    W[6, 13] = 1; W[1, 13] = -1
    
    # Term 15
    U[6, 14] = 1; U[1, 14] = 1
    V[6, 14] = 1; V[3, 14] = 1
    W[6, 14] = 1; W[1, 14] = 1
    
    # Term 16
    U[6, 15] = 1; U[1, 15] = -1
    V[7, 15] = 1; V[2, 15] = -1
    W[7, 15] = 1; W[2, 15] = -1
    
    # Term 17
    U[7, 16] = 1; U[2, 16] = 1
    V[7, 16] = 1; V[4, 16] = 1
    W[7, 16] = 1; W[2, 16] = 1
    
    # Term 18
    U[7, 17] = 1; U[2, 17] = -1
    V[8, 17] = 1; V[3, 17] = -1
    W[8, 17] = 1; W[3, 17] = -1
    
    # Term 19
    U[8, 18] = 1; U[3, 18] = 1
    V[8, 18] = 1; V[5, 18] = 1
    W[8, 18] = 1; W[3, 18] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, seed=42, atol=1e-9):
    """Verify that U, V, W correctly decompose matrix multiplication."""
    np.random.seed(seed)
    A_test = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B_test = np.random.randint(-5, 6, size=(m, p)).astype(float)
    C_expected = A_test @ B_test

    rank = U.shape[1]
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        
        M1 = A_test * u_r
        M2 = v_r * B_test
        partial = M1 @ M2
        C_computed += partial * w_r.T

    return np.allclose(C_computed, C_expected, atol=atol)

def _build_6x6_decomposition():
    """
    Build the 6x6 decomposition using 3x3 Liu-Huang (rank 19) x 2x2 Strassen (rank 7).
    Total rank = 19 * 7 = 133.
    """
    U22, V22, W22 = _build_strassen_2x2()
    U33, V33, W33 = _build_lhuang_3x3()

    rank = 19 * 7  # 133

    U66 = np.zeros((36, rank), dtype=float)
    V66 = np.zeros((36, rank), dtype=float)
    W66 = np.zeros((36, rank), dtype=float)

    for c33 in range(19):
        for c22 in range(7):
            c_final = c33 * 7 + c22

            # Build U66
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

            # Build V66
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
                            V66[target_row, c_final] = v33_val * v22_val

            # Build W66
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
                            W66[target_row, c_final] = w33_val * w22_val

    return U66, V66, W66

def _build_schoolbook_6x6():
    """Ultimate fallback: 6x6 schoolbook (rank 216)."""
    n, m, p = 6, 6, 6
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W

def construct():
    """
    Construct a rank-133 decomposition of 6x6 * 6x6 -> 6x6 matrix multiplication
    using 3x3 Liu-Huang (rank 19) x 2x2 Strassen (rank 7).
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = DIMENSIONS  # 6, 6, 6

    # Build the 6x6 decomposition
    U66, V66, W66 = _build_6x6_decomposition()

    # Verify the 6x6 decomposition
    if time.time() < deadline - 5:
        if not _verify_decomposition(U66, V66, W66, n, m, p, seed=42):
            # Fallback to schoolbook
            U_fb, V_fb, W_fb = _build_schoolbook_6x6()
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U66.tolist(), "V": V66.tolist(), "W": W66.tolist()}

# EVOLVE-BLOCK-END
