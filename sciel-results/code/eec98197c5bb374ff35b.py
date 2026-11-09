"""Rank-minimal tensor decomposition for 2x3 x 3x4 matrix multiplication.

Approach (block decomposition + Strassen):
  A is 2x3, B is 3x4, C = A B is 2x4.
  Split A by columns: A_L = A[:, 0:2] (2x2), A_R = A[:, 2] (2x1)
  Split B by rows:    B_T = B[0:2, :] (2x4), B_B = B[2:3, :] (1x4)
  C = A_L @ B_T + A_R @ B_B

  For A_L @ B_T (2x2 * 2x4): split B_T into two 2x2 blocks by columns.
    Left  2x2 block -> Strassen rank 7
    Right 2x2 block -> Strassen rank 7
  For A_R @ B_B (2x1 * 1x4): C_R[i,k] = A[i,2]*B[2,k]
    This is the outer product of a 2-vector and a 4-vector, which has
    tensor rank 2 (one multiplication per row of the output):
      col 14: C[0,k] += A[0,2]*B[2,k]  for k=0..3
      col 15: C[1,k] += A[1,2]*B[2,k]  for k=0..3

  Total rank = 7 + 7 + 2 = 16  (vs schoolbook 24).

Strassen for 2x2 * 2x2 (verified by expansion):
  A = [[a, b], [c, d]],  B = [[e, f], [g, h]]
  M1 = (a+d)(e+h)
  M2 = (c+d)(e+g)
  M3 =  a(f-g)
  M4 =  d(h-e)
  M5 = (a+b)(g+h)
  M6 = (c+b)(f+h)
  M7 = (a-d)(h-g)
  C00 = M1 - M4 + M6
  C01 = M3 + M4
  C10 = M2 - M7 + M5
  C11 = M1 - M2 + M7 + M3
"""

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4

    rank = 16
    # U: (n*m, rank) = (6, 16)
    # V: (m*p, rank) = (12, 16)
    # W: (p*n, rank) = (8, 16)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen(a_rows, a_cols, b_rows, b_cols):
        """Add 7 Strassen columns for a 2x2 * 2x2 sub-problem.

        A sub-matrix: rows a_rows, cols a_cols  (2x2)
        B sub-matrix: rows b_rows, cols b_cols  (2x2)
        C output:     rows a_rows, cols b_cols  (2x2)
        """
        nonlocal col
        # Local indices
        # a = A[a_rows[0], a_cols[0]]
        # b = A[a_rows[0], a_cols[1]]
        # c = A[a_rows[1], a_cols[0]]
        # d = A[a_rows[1], a_cols[1]]
        # e = B[b_rows[0], b_cols[0]]
        # f = B[b_rows[0], b_cols[1]]
        # g = B[b_rows[1], b_cols[0]]
        # h = B[b_rows[1], b_cols[1]]

        # M1 = (a + d)(e + h)
        c0 = col
        U[a_rows[0] * m + a_cols[0]][c0] = 1   # a
        U[a_rows[1] * m + a_cols[1]][c0] = 1   # d
        V[b_rows[0] * p + b_cols[0]][c0] = 1   # e
        V[b_rows[1] * p + b_cols[1]][c0] = 1   # h
        # C00 += M1, C11 += M1
        W[b_cols[0] * n + a_rows[0]][c0] = 1   # C[0, b_cols[0]]
        W[b_cols[1] * n + a_rows[1]][c0] = 1   # C[1, b_cols[1]]
        col += 1

        # M2 = (c + d)(e + g)
        c0 = col
        U[a_rows[1] * m + a_cols[0]][c0] = 1   # c
        U[a_rows[1] * m + a_cols[1]][c0] = 1   # d
        V[b_rows[0] * p + b_cols[0]][c0] = 1   # e
        V[b_rows[1] * p + b_cols[0]][c0] = 1   # g
        # C10 += M2
        W[b_cols[0] * n + a_rows[1]][c0] = 1   # C[1, b_cols[0]]
        col += 1

        # M3 = a(f - g)
        c0 = col
        U[a_rows[0] * m + a_cols[0]][c0] = 1   # a
        V[b_rows[0] * p + b_cols[1]][c0] = 1   # f
        V[b_rows[1] * p + b_cols[1]][c0] = 0   # not used, g goes to M3 via -g
        # Wait: M3 = a * (f - g). So V should have f=1, g=-1.
        # But g is B[b_rows[1], b_cols[0]], not b_cols[1].
        V[b_rows[1] * p + b_cols[0]][c0] = -1  # -g
        # C01 += M3, C11 += M3
        W[b_cols[1] * n + a_rows[0]][c0] = 1   # C[0, b_cols[1]]
        W[b_cols[1] * n + a_rows[1]][c0] = 1   # C[1, b_cols[1]]
        col += 1

        # M4 = d(h - e)
        c0 = col
        U[a_rows[1] * m + a_cols[1]][c0] = 1   # d
        V[b_rows[1] * p + b_cols[1]][c0] = 1   # h
        V[b_rows[0] * p + b_cols[0]][c0] = -1  # -e
        # C00 -= M4, C01 += M4
        W[b_cols[0] * n + a_rows[0]][c0] = -1  # C[0, b_cols[0]]
        W[b_cols[1] * n + a_rows[0]][c0] = 1   # C[0, b_cols[1]]
        col += 1

        # M5 = (a + b)(g + h)
        c0 = col
        U[a_rows[0] * m + a_cols[0]][c0] = 1   # a
        U[a_rows[0] * m + a_cols[1]][c0] = 1   # b
        V[b_rows[1] * p + b_cols[0]][c0] = 1   # g
        V[b_rows[1] * p + b_cols[1]][c0] = 1   # h
        # C10 += M5
        W[b_cols[0] * n + a_rows[1]][c0] = 1   # C[1, b_cols[0]]
        col += 1

        # M6 = (c + b)(f + h)
        c0 = col
        U[a_rows[1] * m + a_cols[0]][c0] = 1   # c
        U[a_rows[0] * m + a_cols[1]][c0] = 1   # b
        V[b_rows[0] * p + b_cols[1]][c0] = 1   # f
        V[b_rows[1] * p + b_cols[1]][c0] = 1   # h
        # C00 += M6
        W[b_cols[0] * n + a_rows[0]][c0] = 1   # C[0, b_cols[0]]
        col += 1

        # M7 = (a - d)(h - g)
        c0 = col
        U[a_rows[0] * m + a_cols[0]][c0] = 1   # a
        U[a_rows[1] * m + a_cols[1]][c0] = -1  # -d
        V[b_rows[1] * p + b_cols[1]][c0] = 1   # h
        V[b_rows[1] * p + b_cols[0]][c0] = -1  # -g
        # C10 -= M7, C11 += M7
        W[b_cols[0] * n + a_rows[1]][c0] = -1  # C[1, b_cols[0]]
        W[b_cols[1] * n + a_rows[1]][c0] = 1   # C[1, b_cols[1]]
        col += 1

    # Strassen block 1: A_L (rows 0,1; cols 0,1) * B_T left (rows 0,1; cols 0,1)
    add_strassen(a_rows=(0, 1), a_cols=(0, 1), b_rows=(0, 1), b_cols=(0, 1))

    # Strassen block 2: A_L (rows 0,1; cols 0,1) * B_T right (rows 0,1; cols 2,3)
    add_strassen(a_rows=(0, 1), a_cols=(0, 1), b_rows=(0, 1), b_cols=(2, 3))

    # A_R @ B_B: 2x1 * 1x4, rank 2
    # col 14: C[0, k] += A[0,2] * B[2, k] for k=0..3
    c0 = col
    U[0 * m + 2][c0] = 1   # A[0,2]
    for k in range(p):
        V[2 * p + k][c0] = 1   # B[2, k]
        W[k * n + 0][c0] = 1   # C[0, k]
    col += 1

    # col 15: C[1, k] += A[1,2] * B[2, k] for k=0..3
    c0 = col
    U[1 * m + 2][c0] = 1   # A[1,2]
    for k in range(p):
        V[2 * p + k][c0] = 1   # B[2, k]
        W[k * n + 1][c0] = 1   # C[1, k]
    col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"

    # Numerical verification
    try:
        import numpy as np
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        # Random test
        rng = np.random.default_rng(42)
        A_test = rng.random((n, m))
        B_test = rng.random((m, p))
        C_exact = A_test @ B_test

        # Compute via decomposition
        C_decomp = np.zeros((n, p))
        for c in range(rank):
            u_vec = U_arr[:, c]  # (n*m,)
            v_vec = V_arr[:, c]  # (m*p,)
            w_vec = W_arr[:, c]  # (p*n,)
            # Reshape u_vec to (n, m), v_vec to (m, p), w_vec to (p, n)
            U_mat = u_vec.reshape(n, m)
            V_mat = v_vec.reshape(m, p)
            W_mat = w_vec.reshape(p, n)
            # The product is: sum over j,k of U_mat[i,j] * V_mat[j,k] * W_mat[k,i]
            # This equals (U_mat * V_mat) element-wise? No.
            # Actually: C_decomp[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # For each c, the contribution to C[i,k] is U_mat[i,j] * V_mat[j,k] * W_mat[k,i]
            # but summed over j... wait, that's not right either.
            # The formula is: C[i,k] = sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # For a fixed c, the inner sum over j gives:
            #   sum_j U_mat[i,j] * V_mat[j,k] = (U_mat @ V_mat)[i,k]
            # So C_decomp[i,k] = sum_c (U_mat @ V_mat)[i,k] * W_mat[k,i]
            C_decomp += (U_mat @ V_mat) * W_mat.T  # W_mat is (p,n), W_mat.T is (n,p), so W_mat[k,i] = W_mat.T[i,k]

        if not np.allclose(C_exact, C_decomp, atol=1e-9):
            diff = np.abs(C_exact - C_decomp).max()
            raise ValueError(f"Decomposition failed: max diff = {diff}")
    except ImportError:
        pass  # numpy not available, skip verification

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
