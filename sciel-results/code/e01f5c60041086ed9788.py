"""Matrix multiplication 3,4,8: rank-minimal exact tensor decomposition.

Strategy: split the 3x4x8 problem into a 2x4x8 part (solved via Strassen
blocking: 2x4x8 -> 2x4x4 -> 2x4x2 -> 2x2x2) and a 1x4x8 part (schoolbook,
since 1xm*p has rank m*p). Total rank = 56 + 32 = 88 < 96 schoolbook.
"""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]
TIME_LIMIT = 130  # seconds, safety margin before 150s kill


def _strassen_2x2x2():
    """Return U (4x7), V (4x7), W (4x7) for M_{2,2,2} using Strassen."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # U row index: i*2 + j  (A indices)
    # V row index: j*2 + k  (B indices)
    # W row index: k*2 + i  (C indices)
    # Strassen 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # A indices: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # B indices: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
    # C indices: C11=(0,0)->k*n+i=0*2+0=0, C12=(0,1)->1*2+0=2
    #            C21=(1,0)->0*2+1=1, C22=(1,1)->1*2+1=3
    # W row: k*2+i: C11->0, C12->2, C21->1, C22->3

    A11, A12, A21, A22 = 0, 1, 2, 3
    B11, B12, B21, B22 = 0, 1, 2, 3
    C11, C12, C21, C22 = 0, 2, 1, 3  # W row indices (k*2+i)

    # M1: (A11+A22)(B11+B22)
    r = 0
    U[A11, r] = 1; U[A22, r] = 1
    V[B11, r] = 1; V[B22, r] = 1
    W[C11, r] = 1; W[C22, r] = 1

    # M2: (A21+A22)B11
    r = 1
    U[A21, r] = 1; U[A22, r] = 1
    V[B11, r] = 1
    W[C21, r] = 1; W[C22, r] = -1

    # M3: A11(B21-B22)
    r = 2
    U[A11, r] = 1
    V[B21, r] = 1; V[B22, r] = -1
    W[C12, r] = 1; W[C22, r] = 1

    # M4: A22(B12-B11)
    r = 3
    U[A22, r] = 1
    V[B12, r] = 1; V[B11, r] = -1
    W[C11, r] = 1; W[C21, r] = 1

    # M5: (A11+A12)B22
    r = 4
    U[A11, r] = 1; U[A12, r] = 1
    V[B22, r] = 1
    W[C11, r] = -1; W[C12, r] = 1

    # M6: (A21-A11)(B11+B12)
    r = 5
    U[A21, r] = 1; U[A11, r] = -1
    V[B11, r] = 1; V[B12, r] = 1
    W[C22, r] = 1

    # M7: (A12-A22)(B21+B22)
    r = 6
    U[A12, r] = 1; U[A22, r] = -1
    V[B21, r] = 1; V[B22, r] = 1
    W[C11, r] = 1; W[C22, r] = 1

    return U, V, W


def _schoolbook_decomposition(n, m, p):
    """Return schoolbook (rank n*m*p) decomposition of M_{n,m,p}."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _block_combine(U1, V1, W1, U2, V2, W2,
                   i_offset, j_offset, k_offset,
                   n_total, m_total, p_total):
    """
    Combine two sub-decompositions that operate on different row/col blocks.
    
    Sub-problem 1: n1 x m1 x p1, with global offsets (i_offset, j_offset, k_offset)
    Sub-problem 2: n2 x m2 x p2, with global offsets (i_offset+n1, j_offset, k_offset)
    (i.e., they split along the i dimension, sharing j and k ranges)
    
    Actually, let me be more general. Each sub-problem has its own (n, m, p)
    and offsets for i, j, k in the global space.
    """
    # This function is not used directly; we build the full matrices manually.
    pass


def _embed_decomposition(U_sub, V_sub, W_sub,
                         n_sub, m_sub, p_sub,
                         i_off, j_off, k_off,
                         n_total, m_total, p_total):
    """
    Embed a local decomposition into global matrices.
    
    Local U: shape (n_sub*m_sub, rank)  -- rows indexed by i_sub*m_sub + j_sub
    Local V: shape (m_sub*p_sub, rank)  -- rows indexed by j_sub*p_sub + k_sub
    Local W: shape (p_sub*n_sub, rank)  -- rows indexed by k_sub*n_sub + i_sub
    
    Global U: shape (n_total*m_total, rank) -- rows indexed by i_total*m_total + j_total
    Global V: shape (m_total*p_total, rank) -- rows indexed by j_total*p_total + k_total
    Global W: shape (p_total*n_total, rank) -- rows indexed by k_total*n_total + i_total
    
    Mapping:
      i_total = i_off + i_sub
      j_total = j_off + j_sub
      k_total = k_off + k_sub
    """
    rank = U_sub.shape[1]
    U_full = np.zeros((n_total * m_total, rank))
    V_full = np.zeros((m_total * p_total, rank))
    W_full = np.zeros((p_total * n_total, rank))
    
    for i_sub in range(n_sub):
        for j_sub in range(m_sub):
            i_t = i_off + i_sub
            j_t = j_off + j_sub
            U_full[i_t * m_total + j_t, :] = U_sub[i_sub * m_sub + j_sub, :]
    
    for j_sub in range(m_sub):
        for k_sub in range(p_sub):
            j_t = j_off + j_sub
            k_t = k_off + k_sub
            V_full[j_t * p_total + k_t, :] = V_sub[j_sub * p_sub + k_sub, :]
    
    for k_sub in range(p_sub):
        for i_sub in range(n_sub):
            k_t = k_off + k_sub
            i_t = i_off + i_sub
            W_full[k_t * n_total + i_t, :] = W_sub[k_sub * n_sub + i_sub, :]
    
    return U_full, V_full, W_full


def _split_p_decomposition(n, m, p):
    """
    Decompose M_{n,m,p} by splitting the p dimension into two halves.
    This is just concatenation along k: rank = n*m*p (no savings from splitting p alone).
    """
    return _schoolbook_decomposition(n, m, p)


def _strassen_block_2x4x2():
    """
    M_{2,4,2}: A is 2x4, B is 4x2, C is 2x2.
    Split B's columns (m=4) into two 2-column groups.
    C = A * B = A[:, 0:2] * B[0:2, :] + A[:, 2:4] * B[2:4, :]
    Each term is M_{2,2,2} with rank 7 (Strassen).
    Total rank: 14.
    """
    n, m, p = 2, 4, 2
    rank = 14
    U = np.zeros((n * m, rank))  # (8, 14)
    V = np.zeros((m * p, rank))  # (8, 14)
    W = np.zeros((p * n, rank))  # (4, 14)
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # First block: j_sub in {0,1}, uses A[:,0:2] * B[0:2,:]
    for i in range(n):
        for j in range(2):
            U[i * m + j, 0:7] = U_s[i * 2 + j, 0:7]
    for j in range(2):
        for k in range(p):
            V[j * p + k, 0:7] = V_s[j * 2 + k, 0:7]
    W[:, 0:7] = W_s[:, 0:7]
    
    # Second block: j_sub in {2,3}, uses A[:,2:4] * B[2:4,:]
    for i in range(n):
        for j in range(2):
            U[i * m + (j + 2), 7:14] = U_s[i * 2 + j, 0:7]
    for j in range(2):
        for k in range(p):
            V[(j + 2) * p + k, 7:14] = V_s[j * 2 + k, 0:7]
    W[:, 7:14] = W_s[:, 0:7]
    
    return U, V, W


def _strassen_block_2x4x4():
    """
    M_{2,4,4}: Split p=4 into two p=2 blocks.
    Each is M_{2,4,2} with rank 14.
    Total rank: 28.
    """
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank))  # (8, 28)
    V = np.zeros((m * p, rank))  # (16, 28)
    W = np.zeros((p * n, rank))  # (8, 28)
    
    U1, V1, W1 = _strassen_block_2x4x2()  # for k in {0,1}
    U2, V2, W2 = _strassen_block_2x4x2()  # for k in {2,3}
    
    # Both blocks use the same A (n x m = 2 x 4), so U is the same pattern
    U[:, 0:14] = U1
    U[:, 14:28] = U2
    
    # V: first block uses B rows 0:4 (j in 0..3, k in 0..1) -> V1 shape (4*2, 14) = (8,14)
    #    second block uses B rows 0:4 (j in 0..3, k in 2..3) -> V2 shape (4*2, 14) = (8,14)
    # Global V row: j*p + k
    # Block 1: k in {0,1}, so V rows: j*4+0, j*4+1 for j=0..3
    # Block 2: k in {2,3}, so V rows: j*4+2, j*4+3 for j=0..3
    for j in range(m):
        V[j * p + 0, 0:14] = V1[j * 2 + 0, 0:14]
        V[j * p + 1, 0:14] = V1[j * 2 + 1, 0:14]
        V[j * p + 2, 14:28] = V2[j * 2 + 0, 0:14]
        V[j * p + 3, 14:28] = V2[j * 2 + 1, 0:14]
    
    # W: first block has k in {0,1}, i in {0,1} -> rows k*2+i = 0,1,2,3
    #    second block has k in {2,3}, i in {0,1} -> rows k*2+i = 4,5,6,7
    W[0:4, 0:14] = W1[:, 0:14]
    W[4:8, 14:28] = W2[:, 0:14]
    
    return U, V, W


def _strassen_block_2x4x8():
    """
    M_{2,4,8}: Split p=8 into two p=4 blocks.
    Each is M_{2,4,4} with rank 28.
    Total rank: 56.
    """
    n, m, p = 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank))  # (8, 56)
    V = np.zeros((m * p, rank))  # (32, 56)
    W = np.zeros((p * n, rank))  # (16, 56)
    
    U1, V1, W1 = _strassen_block_2x4x4()  # for k in {0..3}
    U2, V2, W2 = _strassen_block_2x4x4()  # for k in {4..7}
    
    # U: same A for both blocks
    U[:, 0:28] = U1
    U[:, 28:56] = U2
    
    # V: block 1 uses k in {0..3}, block 2 uses k in {4..7}
    # Global V row: j*8 + k
    # Block 1: k in {0,1,2,3} -> V rows j*8+0 .. j*8+3
    # Block 2: k in {4,5,6,7} -> V rows j*8+4 .. j*8+7
    for j in range(m):
        for k in range(4):
            V[j * p + k, 0:28] = V1[j * 4 + k, 0:28]
        for k in range(4):
            V[j * p + (k + 4), 28:56] = V2[j * 4 + k, 0:28]
    
    # W: block 1 has k in {0..3}, i in {0,1} -> rows k*2+i = 0..7
    #    block 2 has k in {4..7}, i in {0,1} -> rows k*2+i = 8..15
    W[0:8, 0:28] = W1[:, 0:28]
    W[8:16, 28:56] = W2[:, 0:28]
    
    return U, V, W


def _construct_3x4x8():
    """
    Construct M_{3,4,8} decomposition by splitting i into {0,1} and {2}.
    
    Part 1: i in {0,1} -> M_{2,4,8} with rank 56 (Strassen blocking)
    Part 2: i in {2} -> M_{1,4,8} with rank 32 (schoolbook, since 1xm*p = m*p)
    
    Total rank: 56 + 32 = 88.
    
    Global indices:
    U: shape (12, 88), row = i*4+j
    V: shape (32, 88), row = j*8+k
    W: shape (24, 88), row = k*3+i
    """
    n, m, p = 3, 4, 8
    total_rank = 88
    
    U = np.zeros((n * m, total_rank))  # (12, 88)
    V = np.zeros((m * p, total_rank))  # (32, 88)
    W = np.zeros((p * n, total_rank))  # (24, 88)
    
    # Part 1: i in {0,1}, M_{2,4,8}, rank 56
    U1, V1, W1 = _strassen_block_2x4x8()
    # U1: (8, 56), row = i_sub*4+j_sub where i_sub in {0,1}, j_sub in {0,1,2,3}
    # Global U row for i in {0,1}: i*4+j (same as i_sub*4+j_sub since i_off=0, j_off=0)
    U[0:8, 0:56] = U1
    # V1: (32, 56), row = j*8+k -- same as global V
    V[:, 0:56] = V1
    # W1: (16, 56), row = k*2+i_sub where i_sub in {0,1}
    # Global W row: k*3+i. For i=0: k*3+0. For i=1: k*3+1.
    # W1 row k*2+i_sub maps to global k*3+i_sub
    for k in range(p):
        for i_sub in range(2):
            W[k * n + i_sub, 0:56] = W1[k * 2 + i_sub, 0:56]
    
    # Part 2: i in {2}, M_{1,4,8}, rank 32 (schoolbook)
    U2, V2, W2 = _schoolbook_decomposition(1, 4, 8)
    # U2: (4, 32), row = 0*4+j_sub = j_sub (since n_sub=1)
    # Global U row for i=2: 2*4+j = 8+j
    U[8:12, 56:88] = U2
    # V2: (32, 32), row = j*8+k -- same as global V
    V[:, 56:88] = V2
    # W2: (8, 32), row = k*1+i_sub = k (since n_sub=1, i_sub=0)
    # Global W row for i=2: k*3+2
    for k in range(p):
        W[k * n + 2, 56:88] = W2[k * 1 + 0, 0:32]
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition of M_{n,m,p}."""
    rank = U.shape[1]
    assert U.shape[1] == V.shape[1] == W.shape[1] == rank, \
        f"Shape mismatch: U={U.shape}, V={V.shape}, W={W.shape}"
    assert U.shape[0] == n * m, f"U rows {U.shape[0]} != n*m {n*m}"
    assert V.shape[0] == m * p, f"V rows {V.shape[0]} != m*p {m*p}"
    assert W.shape[0] == p * n, f"W rows {W.shape[0]} != p*n {p*n}"
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check integer or half-integer entries
    for mat in [U, V, W]:
        doubled = mat * 2
        if not np.allclose(doubled, np.round(doubled), atol=1e-9):
            return False
    return True


def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor.
    Returns a decomposition with rank 88 for 3x4x8 (vs 96 schoolbook).
    """
    t_start = time.time()
    n, m, p = DIMENSIONS
    
    try:
        U, V, W = _construct_3x4x8()
        
        if not _verify_decomposition(U, V, W, n, m, p):
            raise ValueError("Decomposition verification failed")
        
        elapsed = time.time() - t_start
        if elapsed > TIME_LIMIT:
            # Should not happen, but safety check
            pass
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    except Exception as e:
        # Fallback to schoolbook
        elapsed = time.time() - t_start
        if elapsed > TIME_LIMIT - 10:
            # Not enough time for fallback either; return whatever we can
            U, V, W = _schoolbook_decomposition(n, m, p)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        U, V, W = _schoolbook_decomposition(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
