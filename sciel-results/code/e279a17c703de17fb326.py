"""
Design: Kronecker product of 3x3 schoolbook (rank 27) and 2x2 Strassen (rank 7),
embedded into 6x6. Total rank = 189.

The 6x6 matrices are partitioned into 3x3 blocks of 2x2.
Each 2x2 block multiplication uses Strassen (rank 7).
Each 3x3 block multiplication uses schoolbook (rank 27).
Total rank: 27 * 7 = 189.

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
    
    # Indices for A, B, C are flattened row-major: 00, 01, 10, 11
    # C indices in W follow W[k*2+i] convention:
    # W[0]=C00, W[1]=C01, W[2]=C10, W[3]=C11
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    # C00 += M1, C11 += M1
    W[0, 0] += 1; W[3, 0] += 1
    
    # M2 = (A10+A11)B00
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    # C10 += M2, C11 -= M2
    W[2, 1] += 1; W[3, 1] -= 1
    
    # M3 = A00(B01-B11)
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] -= 1
    # C01 += M3, C11 += M3
    W[1, 2] += 1; W[3, 2] += 1
    
    # M4 = A11(B10-B00)
    U[3, 3] += 1
    V[2, 3] += 1; V[0, 3] -= 1
    # C00 -= M4, C10 += M4
    W[0, 3] -= 1; W[2, 3] += 1
    
    # M5 = (A00+A01)B11
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    # C00 += M5, C01 += M5
    W[0, 4] += 1; W[1, 4] += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    # C11 += M6
    W[3, 5] += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    # C00 -= M7
    W[0, 6] -= 1
    
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
    # Vectorized verification: C = sum_r (A * U_r) @ (V_r * B) * W_r^T
    # Reshape U, V, W to (rank, n, m), (rank, m, p), (rank, p, n)
    U_r = U.T.reshape(rank, n, m)  # (rank, n, m)
    V_r = V.T.reshape(rank, m, p)  # (rank, m, p)
    W_r = W.T.reshape(rank, p, n)  # (rank, p, n)
    
    # A_test * U_r: (n, m) * (rank, n, m) -> (rank, n, m) via broadcasting
    M1 = A_test[None, :, :] * U_r  # (rank, n, m)
    M2 = V_r * B_test[None, :, :]  # (rank, m, p)
    
    # partial_r = M1[r] @ M2[r] for each r
    # Use einsum: partial[r, i, k] = sum_j M1[r, i, j] * M2[r, j, k]
    partial = np.einsum('rij,rjk->rik', M1, M2)  # (rank, n, p)
    
    # C_computed[i, k] = sum_r partial[r, i, k] * W_r[r, k, i]
    # W_r is (rank, p, n), so W_r[r, k, i]
    C_computed = np.einsum('rik,rki->ik', partial, W_r)

    return np.allclose(C_computed, C_expected, atol=atol)


def _build_6x6_decomposition():
    """
    Build the 6x6 decomposition using 3x3 schoolbook x 2x2 Strassen.
    Total rank = 27 * 7 = 189.
    Uses Kronecker product logic with index permutations.
    """
    U22, V22, W22 = _build_strassen_2x2()
    U33, V33, W33 = _build_schoolbook_3x3()

    # Kronecker product of the factor matrices
    # U_kron has shape (9*4, 27*7) = (36, 189)
    # Row index in kron: r33 * 4 + r22
    U_kron = np.kron(U33, U22)
    V_kron = np.kron(V33, V22)
    W_kron = np.kron(W33, W22)

    U66 = np.zeros_like(U_kron)
    V66 = np.zeros_like(V_kron)
    W66 = np.zeros_like(W_kron)

    # Permute rows to match target index ordering
    # Target U row: I*6 + J, where I = i3*2 + i2, J = j3*2 + j2
    # Kron U row: r33*4 + r22, where r33 = i3*3 + j3, r22 = i2*2 + j2
    
    for I in range(6):
        for J in range(6):
            target_row = I * 6 + J
            i3 = I // 2
            i2 = I % 2
            j3 = J // 2
            j2 = J % 2
            
            r33 = i3 * 3 + j3
            r22 = i2 * 2 + j2
            kron_row = r33 * 4 + r22
            
            U66[target_row, :] = U_kron[kron_row, :]

    # Target V row: J*6 + K, where J = j3*2 + j2, K = k3*2 + k2
    # Kron V row: r33*4 + r22, where r33 = j3*3 + k3, r22 = j2*2 + k2
    for J in range(6):
        for K in range(6):
            target_row = J * 6 + K
            j3 = J // 2
            j2 = J % 2
            k3 = K // 2
            k2 = K % 2
            
            r33 = j3 * 3 + k3
            r22 = j2 * 2 + k2
            kron_row = r33 * 4 + r22
            
            V66[target_row, :] = V_kron[kron_row, :]

    # Target W row: K*6 + I (Trace convention W[k*n+i])
    # K = k3*2 + k2, I = i3*2 + i2
    # Kron W row: r33*4 + r22, where r33 = k3*3 + i3, r22 = k2*2 + i2
    for K in range(6):
        for I in range(6):
            target_row = K * 6 + I
            k3 = K // 2
            k2 = K % 2
            i3 = I // 2
            i2 = I % 2
            
            r33 = k3 * 3 + i3
            r22 = k2 * 2 + i2
            kron_row = r33 * 4 + r22
            
            W66[target_row, :] = W_kron[kron_row, :]

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
    Construct a rank-189 decomposition of 6x6 * 6x6 -> 6x6 matrix multiplication
    using 3x3 schoolbook (rank 27) x 2x2 Strassen (rank 7).
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
