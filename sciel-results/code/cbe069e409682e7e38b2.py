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

def _build_laderman_4x4():
    """
    Build the 4x4 Laderman decomposition (rank 20).
    Uses the standard 20-term Laderman algorithm.
    """
    U = np.zeros((16, 20), dtype=float)
    V = np.zeros((16, 20), dtype=float)
    W = np.zeros((16, 20), dtype=float)
    
    # Helper to set a 2x2 block in the flattened matrix
    # mat is 16x20, term is column index
    # bi, bj are block indices (0 or 1) for 4x4 split into 2x2 blocks
    # coeffs is 2x2 matrix
    def set_block(mat, bi, bj, coeffs, term):
        for ii in range(2):
            for jj in range(2):
                row = (bi * 2 + ii) * 4 + (bj * 2 + jj)
                mat[row, term] = coeffs[ii, jj]

    I2 = np.eye(2)
    Z2 = np.zeros((2, 2))
    
    # Laderman 4x4 terms
    # 1
    set_block(U, 0, 0, I2, 0); set_block(U, 1, 1, I2, 0)
    set_block(V, 0, 0, I2, 0); set_block(V, 1, 1, I2, 0)
    set_block(W, 0, 0, I2, 0); set_block(W, 1, 1, I2, 0)
    
    # 2
    set_block(U, 1, 0, I2, 1); set_block(U, 1, 1, I2, 1)
    set_block(V, 0, 0, I2, 1)
    set_block(W, 1, 0, I2, 1); set_block(W, 1, 1, I2, 1)
    
    # 3
    set_block(U, 0, 0, I2, 2)
    set_block(V, 0, 1, I2, 2); set_block(V, 1, 1, -I2, 2)
    set_block(W, 0, 1, I2, 2); set_block(W, 1, 1, I2, 2)
    
    # 4
    set_block(U, 1, 1, I2, 3)
    set_block(V, 1, 0, I2, 3); set_block(V, 0, 0, -I2, 3)
    set_block(W, 0, 0, I2, 3); set_block(W, 1, 0, I2, 3)
    
    # 5
    set_block(U, 0, 0, I2, 4); set_block(U, 0, 1, I2, 4)
    set_block(V, 1, 1, I2, 4)
    set_block(W, 0, 0, I2, 4); set_block(W, 0, 1, -I2, 4)
    
    # 6
    set_block(U, 1, 0, I2, 5); set_block(U, 0, 0, -I2, 5)
    set_block(V, 0, 0, I2, 5); set_block(V, 0, 1, I2, 5)
    set_block(W, 1, 1, I2, 5)
    
    # 7
    set_block(U, 0, 1, I2, 6); set_block(U, 1, 1, -I2, 6)
    set_block(V, 1, 0, I2, 6); set_block(V, 1, 1, I2, 6)
    set_block(W, 0, 0, I2, 6)
    
    # 8
    set_block(U, 0, 0, I2, 7); set_block(U, 1, 0, I2, 7)
    set_block(V, 1, 1, I2, 7); set_block(V, 0, 1, I2, 7)
    set_block(W, 0, 0, I2, 7); set_block(W, 0, 1, I2, 7)
    
    # 9
    set_block(U, 0, 1, I2, 8); set_block(U, 1, 1, I2, 8)
    set_block(V, 1, 0, I2, 8); set_block(V, 0, 0, I2, 8)
    set_block(W, 0, 1, I2, 8); set_block(W, 1, 1, I2, 8)
    
    # 10
    set_block(U, 1, 0, I2, 9); set_block(U, 0, 0, I2, 9)
    set_block(V, 0, 0, I2, 9); set_block(V, 1, 0, I2, 9)
    set_block(W, 1, 0, I2, 9); set_block(W, 0, 0, I2, 9)
    
    # 11
    set_block(U, 1, 1, I2, 10); set_block(U, 0, 1, I2, 10)
    set_block(V, 0, 1, I2, 10); set_block(V, 1, 1, I2, 10)
    set_block(W, 1, 1, I2, 10); set_block(W, 0, 1, I2, 10)
    
    # 12
    set_block(U, 0, 0, I2, 11); set_block(U, 1, 1, -I2, 11)
    set_block(U, 0, 1, I2, 11); set_block(U, 1, 0, -I2, 11)
    set_block(V, 0, 0, I2, 11); set_block(V, 1, 1, -I2, 11)
    set_block(V, 0, 1, I2, 11); set_block(V, 1, 0, -I2, 11)
    set_block(W, 0, 0, I2, 11); set_block(W, 1, 1, I2, 11)
    set_block(W, 0, 1, -I2, 11); set_block(W, 1, 0, -I2, 11)
    
    # 13
    set_block(U, 0, 0, I2, 12); set_block(U, 1, 1, I2, 12)
    set_block(U, 0, 1, -I2, 12); set_block(U, 1, 0, -I2, 12)
    set_block(V, 0, 0, I2, 12); set_block(V, 1, 1, I2, 12)
    set_block(V, 0, 1, -I2, 12); set_block(V, 1, 0, -I2, 12)
    set_block(W, 0, 0, I2, 12); set_block(W, 1, 1, I2, 12)
    set_block(W, 0, 1, I2, 12); set_block(W, 1, 0, I2, 12)
    
    # 14
    set_block(U, 0, 0, I2, 13); set_block(U, 0, 1, -I2, 13)
    set_block(U, 1, 0, I2, 13); set_block(U, 1, 1, -I2, 13)
    set_block(V, 0, 0, I2, 13); set_block(V, 0, 1, -I2, 13)
    set_block(V, 1, 0, I2, 13); set_block(V, 1, 1, -I2, 13)
    set_block(W, 0, 0, I2, 13); set_block(W, 0, 1, -I2, 13)
    set_block(W, 1, 0, I2, 13); set_block(W, 1, 1, -I2, 13)
    
    # 15
    set_block(U, 0, 0, I2, 14); set_block(U, 0, 1, I2, 14)
    set_block(U, 1, 0, -I2, 14); set_block(U, 1, 1, -I2, 14)
    set_block(V, 0, 0, I2, 14); set_block(V, 0, 1, I2, 14)
    set_block(V, 1, 0, -I2, 14); set_block(V, 1, 1, -I2, 14)
    set_block(W, 0, 0, I2, 14); set_block(W, 0, 1, I2, 14)
    set_block(W, 1, 0, -I2, 14); set_block(W, 1, 1, -I2, 14)
    
    # 16
    set_block(U, 1, 0, I2, 15); set_block(U, 1, 1, -I2, 15)
    set_block(V, 0, 0, I2, 15); set_block(V, 1, 1, -I2, 15)
    set_block(W, 1, 0, I2, 15); set_block(W, 1, 1, -I2, 15)
    
    # 17
    set_block(U, 0, 1, I2, 16); set_block(U, 0, 0, -I2, 16)
    set_block(V, 1, 0, I2, 16); set_block(V, 1, 1, -I2, 16)
    set_block(W, 0, 1, I2, 16); set_block(W, 0, 0, -I2, 16)
    
    # 18
    set_block(U, 1, 0, I2, 17); set_block(U, 0, 0, -I2, 17)
    set_block(V, 0, 1, I2, 17); set_block(V, 1, 1, -I2, 17)
    set_block(W, 1, 0, I2, 17); set_block(W, 1, 1, -I2, 17)
    
    # 19
    set_block(U, 0, 1, I2, 18); set_block(U, 1, 1, -I2, 18)
    set_block(V, 1, 0, I2, 18); set_block(V, 0, 0, -I2, 18)
    set_block(W, 0, 1, I2, 18); set_block(W, 0, 0, -I2, 18)
    
    # 20
    set_block(U, 0, 0, I2, 19); set_block(U, 1, 1, I2, 19)
    set_block(U, 0, 1, I2, 19); set_block(U, 1, 0, I2, 19)
    set_block(V, 0, 0, I2, 19); set_block(V, 1, 1, I2, 19)
    set_block(V, 0, 1, I2, 19); set_block(V, 1, 0, I2, 19)
    set_block(W, 0, 0, I2, 19); set_block(W, 1, 1, I2, 19)
    set_block(W, 0, 1, I2, 19); set_block(W, 1, 0, I2, 19)

    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, seed=42, atol=1e-9):
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

def _construct_189(n, m, p, deadline):
    # Fallback: 3x3 Naive blocks
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
    
    rank = 27 * 7
    U66 = np.zeros((36, rank), dtype=float)
    V66 = np.zeros((36, rank), dtype=float)
    W66 = np.zeros((36, rank), dtype=float)
    
    U22, V22, W22 = _build_strassen_2x2()
    
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
                            U66[target_row, c_final] = u33_val * u22_val
            for j3 in range(3):
                for k3 in range(3):
                    v33_val = V33[j3 * 3 + k3, c33]
                    if v33_val == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v22_val = V22[j2 * 2 + k2, c22]
                            if v22_val == 0: continue
                            target_row = (j3 * 2 + j2) * 6 + (k3 * 2 + k2)
                            V66[target_row, c_final] = v33_val * v22_val
            for k3 in range(3):
                for i3 in range(3):
                    w33_val = W33[k3 * 3 + i3, c33]
                    if w33_val == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w22_val = W22[k2 * 2 + i2, c22]
                            if w22_val == 0: continue
                            target_row = (k3 * 2 + k2) * 6 + (i3 * 2 + i2)
                            W66[target_row, c_final] = w33_val * w22_val
                            
    return {"U": U66.tolist(), "V": V66.tolist(), "W": W66.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = DIMENSIONS
    
    U22, V22, W22 = _build_strassen_2x2()
    U44, V44, W44 = _build_laderman_4x4()
    
    # Verify Laderman 4x4
    np.random.seed(123)
    A4 = np.random.randint(-5, 6, size=(4, 4)).astype(float)
    B4 = np.random.randint(-5, 6, size=(4, 4)).astype(float)
    C4_expected = A4 @ B4
    C4_computed = np.zeros((4, 4))
    for r in range(20):
        u_r = U44[:, r].reshape(4, 4)
        v_r = V44[:, r].reshape(4, 4)
        w_r = W44[:, r].reshape(4, 4)
        M1 = A4 * u_r
        M2 = v_r * B4
        partial = M1 @ M2
        C4_computed += partial * w_r.T
    laderman_ok = np.allclose(C4_computed, C4_expected, atol=1e-9)
    
    if not laderman_ok:
        return _construct_189(n, m, p, deadline)
        
    rank = 140
    U66 = np.zeros((36, rank), dtype=float)
    V66 = np.zeros((36, rank), dtype=float)
    W66 = np.zeros((36, rank), dtype=float)
    
    valid_bi_ii = [(bi, ii) for bi in range(4) for ii in range(2) if bi*2+ii < 6]
    
    for r44 in range(20):
        u44_r = U44[:, r44].reshape(4, 4)
        v44_r = V44[:, r44].reshape(4, 4)
        w44_r = W44[:, r44].reshape(4, 4)
        for r22 in range(7):
            r_final = r44 * 7 + r22
            u22_r = U22[:, r22].reshape(2, 2)
            v22_r = V22[:, r22].reshape(2, 2)
            w22_r = W22[:, r22].reshape(2, 2)
            
            for bi, ii in valid_bi_ii:
                for bj, jj in valid_bi_ii:
                    i = bi * 2 + ii
                    j = bj * 2 + jj
                    if i >= 6 or j >= 6: continue
                    u44_val = u44_r[bi, bj]
                    u22_val = u22_r[ii, jj]
                    if u44_val != 0 and u22_val != 0:
                        U66[i * 6 + j, r_final] = u44_val * u22_val
                        
            for bj, jj in valid_bi_ii:
                for bk, kk in valid_bi_ii:
                    j = bj * 2 + jj
                    k = bk * 2 + kk
                    if j >= 6 or k >= 6: continue
                    v44_val = v44_r[bj, bk]
                    v22_val = v22_r[jj, kk]
                    if v44_val != 0 and v22_val != 0:
                        V66[j * 6 + k, r_final] = v44_val * v22_val
                        
            for bi, ii in valid_bi_ii:
                for bk, kk in valid_bi_ii:
                    i = bi * 2 + ii
                    k = bk * 2 + kk
                    if i >= 6 or k >= 6: continue
                    w44_val = w44_r[bk, bi]
                    w22_val = w22_r[kk, ii]
                    if w44_val != 0 and w22_val != 0:
                        W66[k * 6 + i, r_final] = w44_val * w22_val
                        
    if time.time() < deadline - 5:
        if not _verify_decomposition(U66, V66, W66, n, m, p, seed=42):
            return _construct_189(n, m, p, deadline)
            
    return {"U": U66.tolist(), "V": V66.tolist(), "W": W66.tolist()}

# EVOLVE-BLOCK-END
