import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def _strassen_2x2x2_block(i_off, j_off, k_off, U, V, W, col):
    """
    Fill U, V, W starting at column `col` with the 7 rank-1 Strassen terms
    for a 2x2x2 block at indices (i_off..i_off+1, j_off..j_off+1, k_off..k_off+1).
    Returns the next free column index.

    Strassen (standard):
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22)B11
      M3 = A11(B21-B22)
      M4 = A22(B12-B11)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    Each M is a rank-1 term: u_r ⊗ v_r ⊗ w_r where
      u_r[i*m+j] = coeff of A[i,j] in the first factor
      v_r[j*p+k] = coeff of B[j,k] in the second factor
      w_r[k*n+i] = coeff of C[i,k] contributed (with sign)
    """
    i0, i1 = i_off, i_off + 1
    j0, j1 = j_off, j_off + 1
    k0, k1 = k_off, k_off + 1

    # (u_indices, v_indices, w_indices) for each M, with signs folded into w
    # u_indices: list of (i, j, coeff) for A factors
    # v_indices: list of (j, k, coeff) for B factors
    # w_indices: list of (i, k, coeff) for C output (signed)
    M = []

    # M1 = (A11+A22)(B11+B22)
    M.append(([ (i0,j0,1), (i1,j1,1) ],
              [ (j0,k0,1), (j1,k1,1) ],
              [ (i0,k0,1), (i0,k1,1), (i1,k0,1), (i1,k1,1) ]))
    # M2 = (A21+A22)B11
    M.append(([ (i1,j0,1), (i1,j1,1) ],
              [ (j0,k0,1) ],
              [ (i1,k0,1), (i1,k1,1) ]))
    # M3 = A11(B21-B22)
    M.append(([ (i0,j0,1) ],
              [ (j1,k0,1), (j1,k1,-1) ],
              [ (i0,k0,1), (i0,k1,1) ]))
    # M4 = A22(B12-B11)
    M.append(([ (i1,j1,1) ],
              [ (j0,k1,1), (j0,k0,-1) ],
              [ (i0,k0,1), (i1,k0,1) ]))
    # M5 = (A11+A12)B22
    M.append(([ (i0,j0,1), (i0,j1,1) ],
              [ (j1,k1,1) ],
              [ (i0,k0,-1), (i0,k1,1) ]))
    # M6 = (A21-A11)(B11+B12)
    M.append(([ (i1,j0,1), (i0,j0,-1) ],
              [ (j0,k0,1), (j0,k1,1) ],
              [ (i1,k0,1), (i1,k1,1) ]))
    # M7 = (A12-A22)(B21+B22)
    M.append(([ (i0,j1,1), (i1,j1,-1) ],
              [ (j1,k0,1), (j1,k1,1) ],
              [ (i0,k0,1), (i1,k0,1) ]))

    for u_idx, v_idx, w_idx in M:
        for (i, j, c) in u_idx:
            U[i * 3 + j][col] = c
        for (j, k, c) in v_idx:
            V[j * 6 + k][col] = c
        for (i, k, c) in w_idx:
            W[k * 3 + i][col] = c
        col += 1
    return col

def _schoolbook_block(i_off, j_off, k_off, n_rows, m_cols, p_cols, U, V, W, col):
    """
    Fill U, V, W starting at column `col` with schoolbook rank-1 terms for a
    sub-block: rows i_off..i_off+n_rows-1, cols j_off..j_off+m_cols-1,
    output cols k_off..k_off+p_cols-1.
    Returns the next free column index.
    """
    for di in range(n_rows):
        for dj in range(m_cols):
            for dk in range(p_cols):
                i = i_off + di
                j = j_off + dj
                k = k_off + dk
                U[i * 3 + j][col] = 1
                V[j * 6 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    return col

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    deadline = time.time() + 118  # safety margin

    # Block decomposition:
    # Block 1: rows 0-1, cols 0-1, output cols 0-1  -> 2x2x2 Strassen, rank 7
    # Block 2: all remaining -> schoolbook
    #   Remaining triples: all (i,j,k) not in block 1
    #   Total triples = 3*3*6 = 54
    #   Block 1 covers 2*2*2 = 8 triples
    #   Remaining = 54 - 8 = 46 triples -> rank 46 schoolbook
    # Total rank = 7 + 46 = 53. That's worse than schoolbook 54? No, 53 < 54.
    # Wait, let me recount. The Strassen block replaces 8 rank-1 terms with 7.
    # So total = 7 + (54-8) = 7 + 46 = 53.

    # But we can do better. Let's use a 2x3x3 Strassen-like block.
    # Actually, let's try: rows 0-1 (2 rows), all 3 cols, output cols 0-2 (3 cols)
    # This is a 2x3x3 block. Schoolbook rank = 2*3*3 = 18.
    # Can we do better than 18 for 2x3x3?
    # 2x3x3 can be seen as 2x2x2 (rank 7) + 2x1x2 (rank 4) + 1x3x2 (rank 6)? No...
    # Let me think differently.
    #
    # Alternative: use the fact that 3x3x6 = (2+1)x3x(2+4) or similar.
    # 
    # Let's try: Split A into top 2 rows and bottom 1 row.
    # Split B's output cols into first 2 and last 4.
    # 
    # Block A: rows 0-1, all 3 cols of A, output cols 0-1 of C -> 2x3x2, schoolbook = 12
    # Block B: rows 0-1, all 3 cols of A, output cols 2-5 of C -> 2x3x4, schoolbook = 24
    # Block C: row 2, all 3 cols of A, output cols 0-5 of C -> 1x3x6, schoolbook = 18
    # Total = 12+24+18 = 54. Same as schoolbook.
    #
    # Let's try Strassen on the 2x2x2 sub-blocks:
    # In block A (2x3x2): use Strassen on cols 0-1 of A and output cols 0-1 -> 2x2x2 rank 7
    #   Plus col 2 of A with output cols 0-1 -> 2x1x2 rank 4
    #   Total for block A = 7+4 = 11
    # In block B (2x3x4): use Strassen on cols 0-1 of A and output cols 2-3 -> 2x2x2 rank 7
    #   Plus col 2 of A with output cols 2-3 -> 2x1x2 rank 4
    #   Plus cols 0-1 of A with output cols 4-5 -> 2x2x2 rank 7
    #   Plus col 2 of A with output cols 4-5 -> 2x1x2 rank 4
    #   Total for block B = 7+4+7+4 = 22
    # Block C (1x3x6): schoolbook = 18
    # Total = 11+22+18 = 51. Better!
    #
    # Can we do even better? Let's try to use more Strassen blocks.
    # In block B, we have 4 sub-blocks of size 2x2x2 or 2x1x2.
    # The 2x1x2 blocks can't use Strassen (need 2x2x2).
    #
    # Alternative split: rows 0-1, cols 0-1, output 0-1 -> 2x2x2 Strassen (7)
    # rows 0-1, cols 0-1, output 2-3 -> 2x2x2 Strassen (7)
    # rows 0-1, cols 0-1, output 4-5 -> 2x2x2 Strassen (7)
    # rows 0-1, col 2, output 0-5 -> 2x1x6 schoolbook (12)
    # row 2, cols 0-1, output 0-1 -> 1x2x2 schoolbook (4)
    # row 2, cols 0-1, output 2-3 -> 1x2x2 schoolbook (4)
    # row 2, cols 0-1, output 4-5 -> 1x2x2 schoolbook (4)
    # row 2, col 2, output 0-5 -> 1x1x6 schoolbook (6)
    # Total = 7+7+7+12+4+4+4+6 = 51. Same as before.
    #
    # Let me try yet another approach. What about using the known result that
    # the tensor rank of 3x3x6 multiplication is at most 24? No, that seems too low.
    #
    # Actually, let me just implement the 51-rank version and verify it.

    rank_total = 51
    U = np.zeros((n * m, rank_total))  # (9, 51)
    V = np.zeros((m * p, rank_total))  # (18, 51)
    W = np.zeros((p * n, rank_total))  # (18, 51)

    col = 0
    # Three 2x2x2 Strassen blocks: rows 0-1, cols 0-1, output cols {0-1, 2-3, 4-5}
    for k_off in [0, 2, 4]:
        col = _strassen_2x2x2_block(0, 0, k_off, U, V, W, col)

    # 2x1x6 schoolbook: rows 0-1, col 2, output cols 0-5
    col = _schoolbook_block(0, 2, 0, 2, 1, 6, U, V, W, col)

    # Row 2, cols 0-1, output cols 0-1
    col = _schoolbook_block(2, 0, 0, 1, 2, 2, U, V, W, col)
    # Row 2, cols 0-1, output cols 2-3
    col = _schoolbook_block(2, 0, 2, 1, 2, 2, U, V, W, col)
    # Row 2, cols 0-1, output cols 4-5
    col = _schoolbook_block(2, 0, 4, 1, 2, 2, U, V, W, col)
    # Row 2, col 2, output cols 0-5
    col = _schoolbook_block(2, 2, 0, 1, 1, 6, U, V, W, col)

    assert col == rank_total, f"Column mismatch: {col} vs {rank_total}"

    # Verify: for each (i,k), sum_j A[i,j]*B[j,k] should equal
    # sum_r U[i*m+j]*V[j*p+k]*W[k*n+i] ... wait, that's not right.
    # The tensor decomposition means:
    # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T_{i,j,k} = 1 for all (i,j,k).
    # So we need: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1

    # Verify
    T_recon = np.zeros((n, m, p))
    for r in range(rank_total):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    # T should be all ones
    if not np.allclose(T_recon, 1.0, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Round to integers (entries should already be integers)
    U = np.rint(U).astype(int)
    V = np.rint(V).astype(int)
    W = np.rint(W).astype(int)

    # Re-verify after rounding
    T_recon2 = np.zeros((n, m, p))
    for r in range(rank_total):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon2[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    if not np.allclose(T_recon2, 1.0, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
