"""Find rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 4, 6]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition of M_{n,m,p}."""
    rank = len(U[0])
    # Build the full multiplication tensor
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        # The outer product uc ⊗ vc ⊗ wc should contribute to T
        # T[i*m+j, j*p+k, k*n+i] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
                    # We need to check that the sum over c equals 1
                    # This is expensive; do it differently
        # Better: compute C = A@B symbolically
    # Actually, let's do a numerical check with random matrices
    rng = np.random.default_rng(42)
    for _ in range(3):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_expected = A @ B
        # Compute C using the decomposition
        # C[i,k] = sum_c W[k*n+i, c] * sum_j U[i*m+j, c] * A[i,j] * sum_... 
        # Actually: C[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
        C_actual = np.zeros((n, p))
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        for c in range(rank):
            # u_c is length n*m, v_c is length m*p, w_c is length p*n
            u = U_arr[:, c].reshape(n, m)
            v = V_arr[:, c].reshape(m, p)
            w = W_arr[:, c]  # length p*n, w[k*n+i]
            # contribution: sum_j A[i,j]*u[i,j] * sum_j B[j,k]*v[j,k] * w[k*n+i]
            au = A * u  # n x m
            bv = B * v  # m x p
            # C[i,k] += au[i,j] summed over j? No.
            # C[i,k] = sum_c (sum_j A[i,j] u[i,j]) (sum_j B[j,k] v[j,k]) w[k*n+i]
            au_sum = au.sum(axis=1)  # n-vector
            bv_sum = bv.sum(axis=0)  # p-vector
            for i in range(n):
                for k in range(p):
                    C_actual[i, k] += au_sum[i] * bv_sum[k] * w[k*n+i]
        if not np.allclose(C_actual, C_expected, atol=1e-9):
            return False
    return True

def _strassen_decomp():
    """Return rank-7 decomposition for 2x2 * 2x2 matrix multiplication.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 multiplications:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # For each of the 7 multiplications c:
    # U[i*2+j, c] = coefficient of A[i,j] in the first factor
    # V[j*2+k, c] = coefficient of B[j,k] in the second factor
    # W[k*2+i, c] = coefficient in C[i,k]
    
    # c=0: M1 = (a00+a11)(b00+b11) -> C00 += 1, C11 += 1
    U[0, 0] = 1; U[3, 0] = 1  # a00, a11
    V[0, 0] = 1; V[3, 0] = 1  # b00, b11
    W[0, 0] = 1; W[3, 0] = 1  # C00, C11
    
    # c=1: M2 = (a10+a11)b00 -> C10 += 1, C11 -= 1
    U[2, 1] = 1; U[3, 1] = 1  # a10, a11
    V[0, 1] = 1  # b00
    W[2, 1] = 1; W[3, 1] = -1  # C10, C11
    
    # c=2: M3 = a00(b01-b11) -> C01 += 1, C11 += 1
    U[0, 2] = 1  # a00
    V[1, 2] = 1; V[3, 2] = -1  # b01, b11
    W[1, 2] = 1; W[3, 2] = 1  # C01, C11
    
    # c=3: M4 = a11(b10-b00) -> C00 += 1, C10 += 1
    U[3, 3] = 1  # a11
    V[2, 3] = 1; V[0, 3] = -1  # b10, b00
    W[0, 3] = 1; W[2, 3] = 1  # C00, C10
    
    # c=4: M5 = (a00+a01)b11 -> C00 -= 1, C01 += 1
    U[0, 4] = 1; U[1, 4] = 1  # a00, a01
    V[3, 4] = 1  # b11
    W[0, 4] = -1; W[1, 4] = 1  # C00, C01
    
    # c=5: M6 = (a10-a00)(b00+b01) -> C11 += 1
    U[2, 5] = 1; U[0, 5] = -1  # a10, a00
    V[0, 5] = 1; V[1, 5] = 1  # b00, b01
    W[3, 5] = 1  # C11
    
    # c=6: M7 = (a01-a11)(b10+b11) -> C00 += 1
    U[1, 6] = 1; U[3, 6] = -1  # a01, a11
    V[2, 6] = 1; V[3, 6] = 1  # b10, b11
    W[0, 6] = 1  # C00
    
    return U.tolist(), V.tolist(), W.tolist()

def _embed_decomp(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub, 
                  n_full, m_full, p_full,
                  i_offset, j_offset, k_offset):
    """Embed a sub-decomposition into a larger one.
    U_sub: (n_sub*m_sub, rank), V_sub: (m_sub*p_sub, rank), W_sub: (p_sub*n_sub, rank)
    Returns (U, V, W) for the full size.
    """
    rank = len(U_sub[0])
    U = np.zeros((n_full * m_full, rank))
    V = np.zeros((m_full * p_full, rank))
    W = np.zeros((p_full * n_full, rank))
    
    for i in range(n_sub):
        for j in range(m_sub):
            idx_sub = i * m_sub + j
            idx_full = (i_offset + i) * m_full + (j_offset + j)
            U[idx_full, :] = U_sub[idx_sub, :]
    
    for j in range(m_sub):
        for k in range(p_sub):
            idx_sub = j * p_sub + k
            idx_full = (j_offset + j) * p_full + (k_offset + k)
            V[idx_full, :] = V_sub[idx_sub, :]
    
    for k in range(p_sub):
        for i in range(n_sub):
            idx_sub = k * n_sub + i
            idx_full = (k_offset + k) * n_full + (i_offset + i)
            W[idx_full, :] = W_sub[idx_sub, :]
    
    return U.tolist(), V.tolist(), W.tolist()

def _block_diagonal_concat(decomps, n, m, p):
    """Concatenate multiple decompositions block-diagonally.
    Each decomp is (U, V, W) for some sub-block.
    """
    # We'll just concatenate the columns
    U_parts = []
    V_parts = []
    W_parts = []
    for U, V, W in decomps:
        U_parts.append(np.array(U))
        V_parts.append(np.array(V))
        W_parts.append(np.array(W))
    
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    """Construct a tensor decomposition for 2x4 * 4x6 matrix multiplication."""
    n, m, p = 2, 4, 6
    
    # Strategy: Split the 4x6 B matrix into blocks.
    # Option 1: Split B into three 4x2 blocks: C = A*B1 | A*B2 | A*B3
    # Each A*Bk is 2x2 multiplication, rank 7 each. Total rank = 21.
    # But wait, A is shared. The tensor decomposition for M_{2,4,2} has rank 14 (not 7)
    # because A is 2x4 and B is 4x2.
    # M_{2,4,2}: A is 2x4, B is 4x2, C is 2x2.
    # We can view this as two 2x2 multiplications: C = [A1 A2] * [[B1],[B2]] where A1,A2 are 2x2, B1,B2 are 2x2.
    # C = A1*B1 + A2*B2. This is a sum of two 2x2 multiplications.
    # Rank of M_{2,2,2} is 7. So rank of M_{2,4,2} <= 14.
    # Actually, M_{2,4,2} is isomorphic to M_{2,2,4} by transposing B.
    # M_{2,2,4}: A is 2x2, B is 2x4, C is 2x4. C = A*[B1 B2] = [A*B1 A*B2].
    # This is two independent 2x2 multiplications, so rank = 7+7 = 14.
    # So M_{2,4,2} has rank 14.
    
    # For M_{2,4,6}: A is 2x4, B is 4x6, C is 2x6.
    # Split B into three 4x2 blocks: B = [B1 | B2 | B3], C = [A*B1 | A*B2 | A*B3].
    # Each A*Bk is a 2x2 result from 2x4 * 4x2, which has rank 14.
    # Total rank = 14 * 3 = 42.
    
    # Alternatively, split B into two 4x3 blocks: C = [A*B1 | A*B2], each is 2x3.
    # M_{2,4,3}: A is 2x4, B is 4x3, C is 2x3.
    # This is more complex. Let's stick with the three 4x2 blocks.
    
    # Construct M_{2,4,2} decomposition (rank 14)
    # M_{2,4,2} = M_{2,2,2} ⊕ M_{2,2,2} (split A into two 2x2 blocks)
    # A = [A1 | A2], B = [B1; B2] where A1,A2 are 2x2, B1,B2 are 2x2.
    # C = A1*B1 + A2*B2
    
    # First, get Strassen decomposition for 2x2*2x2
    U_s, V_s, W_s = _strassen_decomp()
    
    # Embed into M_{2,4,2}:
    # A is 2x4: A = [A1 | A2] where A1 is columns 0-1, A2 is columns 2-3
    # B is 4x2: B = [B1; B2] where B1 is rows 0-1, B2 is rows 2-3
    # C is 2x2
    
    # For the first 2x2 block (A1*B1):
    U_1, V_1, W_1 = _embed_decomp(U_s, V_s, W_s, 2, 2, 2, 
                                   2, 4, 2, 
                                   0, 0, 0)
    # For the second 2x2 block (A2*B2):
    U_2, V_2, W_2 = _embed_decomp(U_s, V_s, W_s, 2, 2, 2, 
                                   2, 4, 2, 
                                   0, 2, 0)
    
    # Concatenate for M_{2,4,2}
    U_242, V_242, W_242 = _block_diagonal_concat(
        [(U_1, V_1, W_1), (U_2, V_2, W_2)], 2, 4, 2)
    
    # Now for M_{2,4,6}, split into three M_{2,4,2} blocks:
    # B is 4x6: B = [B1 | B2 | B3] where each Bk is 4x2
    # C is 2x6: C = [C1 | C2 | C3] where each Ck is 2x2
    
    # For the first block (C1 = A*B1, B1 is columns 0-1 of B):
    U_b1, V_b1, W_b1 = _embed_decomp(U_242, V_242, W_242, 2, 4, 2,
                                      2, 4, 6,
                                      0, 0, 0)
    # For the second block (C2 = A*B2, B2 is columns 2-3 of B):
    U_b2, V_b2, W_b2 = _embed_decomp(U_242, V_242, W_242, 2, 4, 2,
                                      2, 4, 6,
                                      0, 0, 2)
    # For the third block (C3 = A*B3, B3 is columns 4-5 of B):
    U_b3, V_b3, W_b3 = _embed_decomp(U_242, V_242, W_242, 2, 4, 2,
                                      2, 4, 6,
                                      0, 0, 4)
    
    # Concatenate all three blocks
    U, V, W = _block_diagonal_concat(
        [(U_b1, V_b1, W_b1), (U_b2, V_b2, W_b2), (U_b3, V_b3, W_b3)],
        2, 4, 6)
    
    rank = len(U[0])
    
    # Verify the decomposition
    if not _verify(U, V, W, 2, 4, 6):
        # Fallback to schoolbook if verification fails
        rank = 48
        U = [[0] * rank for _ in range(8)]
        V = [[0] * rank for _ in range(24)]
        W = [[0] * rank for _ in range(12)]
        column = 0
        for i in range(2):
            for j in range(4):
                for k in range(6):
                    U[i*4+j][column] = 1
                    V[j*6+k][column] = 1
                    W[k*2+i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
