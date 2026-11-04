"""
Design: Kronecker product of 3x3 schoolbook (rank 27) and 2x2 Strassen (rank 7),
embedded into 6x6. Total rank = 189.

The 6x6 matrices are partitioned into 2x2 blocks of 3x3.
The 2x2 block-level multiplication uses Strassen (rank 7).
Each 3x3 block multiplication uses schoolbook (rank 27).
Total rank: 7 * 27 = 189.

Indexing convention:
- A[i,j] stored at row i*6+j
- B[j,k] stored at row j*6+k
- C[i,k] stored at row k*6+i (the official trace convention W[k*n+i, r])
"""
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


def _build_schoolbook_3x3():
    """Build the 3x3 schoolbook decomposition (rank 27)."""
    U = np.zeros((9, 27), dtype=float)
    V = np.zeros((9, 27), dtype=float)
    W = np.zeros((9, 27), dtype=float)

    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j, col] = 1
                V[j * 3 + k, col] = 1
                W[k * 3 + i, col] = 1
                col += 1

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, seed=42, atol=1e-9):
    """Verify that U, V, W correctly decompose matrix multiplication."""
    np.random.seed(seed)
    A_test = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B_test = np.random.randint(-5, 6, size=(m, p)).astype(float)
    C_expected = A_test @ B_test

    rank = U.shape[1]
    U_r = U.T.reshape(rank, n, m)
    V_r = V.T.reshape(rank, m, p)
    W_r = W.T.reshape(rank, p, n)
    
    M1 = A_test[None, :, :] * U_r
    M2 = V_r * B_test[None, :, :]
    partial = np.einsum('rij,rjk->rik', M1, M2)
    C_computed = np.einsum('rik,rki->ik', partial, W_r)

    return np.allclose(C_computed, C_expected, atol=atol)


def _build_6x6_decomposition():
    """
    Build the 6x6 decomposition using 2x2 Strassen x 3x3 schoolbook.
    Total rank = 7 * 27 = 189.
    
    Partition 6x6 into 2x2 blocks of 3x3.
    Block (b_i, b_j) of A has elements at rows (b_i*3 + i)*6 + (b_j*3 + j) for i,j in 0..2
    """
    U22, V22, W22 = _build_strassen_2x2()
    U33, V33, W33 = _build_schoolbook_3x3()

    rank = 7 * 27  # 189

    U66 = np.zeros((36, rank), dtype=float)
    V66 = np.zeros((36, rank), dtype=float)
    W66 = np.zeros((36, rank), dtype=float)

    for c22 in range(7):
        for c33 in range(27):
            c_final = c22 * 27 + c33

            # Build U66: row = (b_i*3 + i)*6 + (b_j*3 + j)
            # U22[b_i*2 + b_j, c22] * U33[i*3 + j, c33]
            for bi in range(2):
                for bj in range(2):
                    u22_val = U22[bi * 2 + bj, c22]
                    if u22_val == 0:
                        continue
                    for i in range(3):
                        for j in range(3):
                            u33_val = U33[i * 3 + j, c33]
                            if u33_val == 0:
                                continue
                            target_row = (bi * 3 + i) * 6 + (bj * 3 + j)
                            U66[target_row, c_final] = u22_val * u33_val

            # Build V66: row = (b_j*3 + j)*6 + (b_k*3 + k)
            # V22[b_j*2 + b_k, c22] * V33[j*3 + k, c33]
            for bj in range(2):
                for bk in range(2):
                    v22_val = V22[bj * 2 + bk, c22]
                    if v22_val == 0:
                        continue
                    for j in range(3):
                        for k in range(3):
                            v33_val = V33[j * 3 + k, c33]
                            if v33_val == 0:
                                continue
                            target_row = (bj * 3 + j) * 6 + (bk * 3 + k)
                            V66[target_row, c_final] = v22_val * v33_val

            # Build W66: row = (b_k*3 + k)*6 + (b_i*3 + i)
            # W22[b_k*2 + b_i, c22] * W33[k*3 + i, c33]
            for bk in range(2):
                for bi in range(2):
                    w22_val = W22[bk * 2 + bi, c22]
                    if w22_val == 0:
                        continue
                    for k in range(3):
                        for i in range(3):
                            w33_val = W33[k * 3 + i, c33]
                            if w33_val == 0:
                                continue
                            target_row = (bk * 3 + k) * 6 + (bi * 3 + i)
                            W66[target_row, c_final] = w22_val * w33_val

    return U66, V66, W66


def _build_schoolbook_6x6():
    """Fallback: 6x6 schoolbook (rank 216)."""
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
    Construct a rank-189 decomposition of 6x6 * 6x6 -> 6x6 matrix multiplication
    using 2x2 Strassen (rank 7) x 3x3 schoolbook (rank 27).
    """
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    U66, V66, W66 = _build_6x6_decomposition()

    if time.time() < deadline - 5:
        if not _verify_decomposition(U66, V66, W66, n, m, p, seed=42):
            U_fb, V_fb, W_fb = _build_schoolbook_6x6()
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U66.tolist(), "V": V66.tolist(), "W": W66.tolist()}

# EVOLVE-BLOCK-END
