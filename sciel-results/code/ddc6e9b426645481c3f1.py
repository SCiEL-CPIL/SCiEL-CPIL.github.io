"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    rank = 39
    U = [[0] * rank for _ in range(n * m)]  # 6 x 39
    V = [[0] * rank for _ in range(m * p)]  # 21 x 39
    W = [[0] * rank for _ in range(p * n)]  # 14 x 39

    col = 0

    def schoolbook_block(i_start, i_end, j_start, j_end, k_start, k_end):
        """Schoolbook for n_block x m_block x p_block subproblem."""
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1

    def strassen_block(j_start, k_start):
        """Strassen for 2x2x2: A[0:2, j_start:j_start+2] * B[j_start:j_start+2, k_start:k_start+2]"""
        nonlocal col
        A00 = 0 * m + (j_start + 0)
        A01 = 0 * m + (j_start + 1)
        A10 = 1 * m + (j_start + 0)
        A11 = 1 * m + (j_start + 1)

        B00 = (j_start + 0) * p + (k_start + 0)
        B01 = (j_start + 0) * p + (k_start + 1)
        B10 = (j_start + 1) * p + (k_start + 0)
        B11 = (j_start + 1) * p + (k_start + 1)

        C00 = (k_start + 0) * n + 0
        C01 = (k_start + 1) * n + 0
        C10 = (k_start + 0) * n + 1
        C11 = (k_start + 1) * n + 1

        # M1 = (A00 + A11)(B00 + B11) -> C00 += M1, C11 += M1
        U[A00][col] += 1
        U[A11][col] += 1
        V[B00][col] += 1
        V[B11][col] += 1
        W[C00][col] += 1
        W[C11][col] += 1
        col += 1

        # M2 = (A10 + A11)B00 -> C10 += M2
        U[A10][col] += 1
        U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1
        col += 1

        # M3 = A00(B01 - B11) -> C01 += M3
        U[A00][col] += 1
        V[B01][col] += 1
        V[B11][col] += -1
        W[C01][col] += 1
        col += 1

        # M4 = A11(B10 - B00) -> C00 += M4, C10 += M4
        U[A11][col] += 1
        V[B10][col] += 1
        V[B00][col] += -1
        W[C00][col] += 1
        W[C10][col] += 1
        col += 1

        # M5 = (A00 + A01)B11 -> C00 -= M5, C01 += M5
        U[A00][col] += 1
        U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] += -1
        W[C01][col] += 1
        col += 1

        # M6 = (A10 - A00)(B00 + B01) -> C11 += M6
        U[A10][col] += 1
        U[A00][col] += -1
        V[B00][col] += 1
        V[B01][col] += 1
        W[C11][col] += 1
        col += 1

        # M7 = (A01 - A11)(B10 + B11) -> C00 += M7
        U[A01][col] += 1
        U[A11][col] += -1
        V[B10][col] += 1
        V[B11][col] += 1
        W[C00][col] += 1
        col += 1

    # Block 1: A[:,0] * B[0,0:2] -> 2x1x2 schoolbook, rank 4
    schoolbook_block(0, 2, 0, 1, 0, 2)

    # Block 2: A[:,1:3] * B[1:3,0:2] -> 2x2x2 Strassen, rank 7
    strassen_block(1, 0)

    # Block 3: A[:,0] * B[0,2:4] -> 2x1x2 schoolbook, rank 4
    schoolbook_block(0, 2, 0, 1, 2, 4)

    # Block 4: A[:,1:3] * B[1:3,2:4] -> 2x2x2 Strassen, rank 7
    strassen_block(1, 2)

    # Block 5: A[:,0] * B[0,4:7] -> 2x1x3 schoolbook, rank 6
    schoolbook_block(0, 2, 0, 1, 4, 7)

    # Block 6: A[:,1:3] * B[1:3,4:7] -> 2x2x3, split as 2x2x2 + 2x2x1
    # Sub-block 6a: 2x2x2 Strassen (k=4,5), rank 7
    strassen_block(1, 4)

    # Sub-block 6b: 2x2x1 schoolbook (k=6), rank 4
    schoolbook_block(0, 2, 1, 3, 6, 7)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
