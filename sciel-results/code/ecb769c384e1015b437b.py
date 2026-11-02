import numpy as np
import time

# EVOLVE-BLOCK-START

def _get_strassen_2x2():
    """
    Returns U, V, W for 2x2 Strassen decomposition.
    U: (4, 7) - A entries (row-major) to 7 terms
    V: (4, 7) - B entries (row-major) to 7 terms
    W: (4, 7) - 7 terms to C entries (row-major? or specific convention?)
    
    Convention check:
    A = [[A00, A01], [A10, A11]]
    B = [[B00, B01], [B10, B11]]
    
    Terms:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    U (A to M):
    M1: A00(1), A11(1) -> [1, 0, 0, 1]
    M2: A10(1), A11(1) -> [0, 0, 1, 1]
    M3: A00(1)       -> [1, 0, 0, 0]
    M4: A11(1)       -> [0, 0, 0, 1]
    M5: A00(1), A01(1)-> [1, 1, 0, 0]
    M6: A10(1), A00(-1)-> [-1, 0, 1, 0]
    M7: A01(1), A11(-1)-> [0, 1, 0, -1]
    
    U is 4x7. Rows are A00, A01, A10, A11. Cols are M1..M7.
    U[0, :] = [1, 0, 1, 0, 1, -1, 0]
    U[1, :] = [0, 0, 0, 0, 1, 0, 1]
    U[2, :] = [0, 1, 0, 0, 0, 1, 0]
    U[3, :] = [1, 1, 0, 1, 0, 0, -1]
    
    V (B to M):
    M1: B00(1), B11(1) -> [1, 0, 0, 1]
    M2: B00(1)         -> [1, 0, 0, 0]
    M3: B01(1), B11(-1)-> [0, 1, 0, -1]
    M4: B10(1), B00(-1)-> [-1, 0, 1, 0]
    M5: B11(1)         -> [0, 0, 0, 1]
    M6: B00(1), B01(1) -> [1, 1, 0, 0]
    M7: B10(1), B11(1) -> [0, 0, 1, 1]
    
    V is 4x7. Rows are B00, B01, B10, B11.
    V[0, :] = [1, 1, 0, -1, 0, 1, 0]
    V[1, :] = [0, 0, 1, 0, 0, 1, 0]
    V[2, :] = [0, 0, 0, 1, 0, 0, 1]
    V[3, :] = [1, 0, -1, 0, 1, 0, 1]
    
    W (M to C):
    C00: M1(1), M4(1), M5(-1), M7(1) -> [1, 0, 0, 1, -1, 0, 1]
    C01: M3(1), M5(1)                 -> [0, 0, 1, 0, 1, 0, 0]
    C10: M2(1), M4(1)                 -> [0, 1, 0, 1, 0, 0, 0]
    C11: M1(1), M2(-1), M3(1), M6(1)  -> [1, -1, 1, 0, 0, 1, 0]
    
    W is 4x7. Rows are C00, C01, C10, C11.
    W[0, :] = [1, 0, 0, 1, -1, 0, 1]
    W[1, :] = [0, 0, 1, 0, 1, 0, 0]
    W[2, :] = [0, 1, 0, 1, 0, 0, 0]
    W[3, :] = [1, -1, 1, 0, 0, 1, 0]
    """
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """
    Verify that U, V, W decompose matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank) - Index k*n + i
    
    Check: For random A, B, does C = A @ B match the decomposition?
    """
    rank = U.shape[1]
    
    # Reshape U to (n, m, rank)
    U_r = U.reshape(n, m, rank)
    # Reshape V to (m, p, rank)
    V_r = V.reshape(m, p, rank)
    # Reshape W to (p, n, rank) -> W[k, i, r]
    W_r = W.reshape(p, n, rank)
    
    # The decomposition computes:
    # C[i, k] = sum_r W[k, i, r] * (sum_j U[i, j, r] * A[i, j]) * (sum_j V[j, k, r] * B[j, k])
    # Wait, the standard form is:
    # S_r = (U_r^T A_vec) * (V_r^T B_vec)
    # C_vec = sum_r W_r * S_r
    
    # Let's use random matrices for verification
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_true = A @ B
    
    # Compute S_r
    # U_r is (n, m, rank). A is (n, m).
    # U_r * A element-wise? No.
    # S_r = sum_{i,j} U[i, j, r] * A[i, j]
    S_U = np.einsum('ijr,ij->r', U_r, A)
    S_V = np.einsum('jkr,jk->r', V_r, B)
    S = S_U * S_V
    
    # C_calc[i, k] = sum_r W[k, i, r] * S[r]
    # W_r is (p, n, rank) -> W[k, i, r]
    C_calc = np.einsum('kir,r->ik', W_r, S)
    
    return bool(np.allclose(C_true, C_calc, atol=atol))

def construct():
    n, m, p = 4, 4, 4
    start_time = time.time()
    deadline = 120.0
    
    # Get 2x2 Strassen factors
    U2, V2, W2 = _get_strassen_2x2()
    
    # Construct 4x4 factors using Kronecker products
    # U4 = U2 (x) U2
    # V4 = V2 (x) V2
    # W4 = W2 (x) W2
    
    # Note on ordering:
    # A is 4x4. Vectorized A is [A00, A01, A10, A11] (blocks).
    # Each block is 2x2, vectorized [a00, a01, a10, a11].
    # Full vector: [A00_vec, A01_vec, A10_vec, A11_vec].
    # U2 acts on the 4 blocks.
    # U4 = kron(U2, U2) acts on the 16 elements.
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Check shapes
    # U2 is 4x7. U4 is 16x49. Correct.
    # W2 is 4x7. W4 is 16x49.
    # W index convention: W[k*n+i, r].
    # In W2, row index is C index (0..3). C is 2x2.
    # C00 (idx 0), C01 (idx 1), C10 (idx 2), C11 (idx 3).
    # In W4, row index is C index (0..15).
    # The kron product W2 (x) W2 produces rows in order:
    # (W2_row0 (x) W2_row0), (W2_row0 (x) W2_row1), ...
    # This corresponds to C_block_00 (inner C00, C01, C10, C11), C_block_01, etc.
    # This matches the row-major order of the 4x4 C matrix.
    
    # Verify
    if _verify_decomposition(U4, V4, W4, n, m, p):
        return {
            "U": U4.tolist(),
            "V": V4.tolist(),
            "W": W4.tolist()
        }
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

# EVOLVE-BLOCK-END
