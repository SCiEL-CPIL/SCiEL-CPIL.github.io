DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    # Use Strassen's 7-multiplication algorithm on 5 independent 2x2 column blocks
    # of the B matrix (2x10), giving rank 7*5 = 35.
    rank = 35
    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35

    # A is 2x2: A[0,0], A[0,1], A[1,0], A[1,1]
    # U indices: 0->A[0,0], 1->A[0,1], 2->A[1,0], 3->A[1,1]
    # B is 2x10: V indices 0..9 are row 0, 10..19 are row 1
    # C is 10x2: W index for C[j, i] is j*n + i = j*2 + i

    # Strassen's 7 products for multiplying two 2x2 matrices:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    for block in range(5):
        # Block uses columns 2*block and 2*block+1 of B
        c0 = 2 * block       # first column of B in this block
        c1 = 2 * block + 1   # second column of B in this block

        # V indices
        v_b00 = c0           # B[0, c0]
        v_b01 = c1           # B[0, c1]
        v_b10 = 10 + c0      # B[1, c0]
        v_b11 = 10 + c1      # B[1, c1]

        # W indices: C row indices are c0 and c1, column indices 0 and 1
        w_c00 = c0 * 2 + 0   # C[c0, 0]
        w_c01 = c0 * 2 + 1   # C[c0, 1]
        w_c10 = c1 * 2 + 0   # C[c1, 0]
        w_c11 = c1 * 2 + 1   # C[c1, 1]

        base = 7 * block

        # M1 = (A00 + A11)(B00 + B11)
        r = base + 0
        U[0][r] += 1
        U[3][r] += 1
        V[v_b00][r] += 1
        V[v_b11][r] += 1
        W[w_c00][r] += 1
        W[w_c11][r] += 1

        # M2 = (A10 + A11)B00
        r = base + 1
        U[2][r] += 1
        U[3][r] += 1
        V[v_b00][r] += 1
        W[w_c10][r] += 1
        W[w_c11][r] += -1

        # M3 = A00(B01 - B11)
        r = base + 2
        U[0][r] += 1
        V[v_b01][r] += 1
        V[v_b11][r] += -1
        W[w_c01][r] += 1
        W[w_c11][r] += 1

        # M4 = A11(B10 - B00)
        r = base + 3
        U[3][r] += 1
        V[v_b10][r] += 1
        V[v_b00][r] += -1
        W[w_c00][r] += 1
        W[w_c10][r] += 1

        # M5 = (A00 + A01)B11
        r = base + 4
        U[0][r] += 1
        U[1][r] += 1
        V[v_b11][r] += 1
        W[w_c00][r] += -1
        W[w_c01][r] += 1

        # M6 = (A10 - A00)(B00 + B01)
        r = base + 5
        U[2][r] += 1
        U[0][r] += -1
        V[v_b00][r] += 1
        V[v_b01][r] += 1
        W[w_c11][r] += 1

        # M7 = (A01 - A11)(B10 + B11)
        r = base + 6
        U[1][r] += 1
        U[3][r] += -1
        V[v_b10][r] += 1
        V[v_b11][r] += 1
        W[w_c00][r] += 1

    return {"U": U, "V": V, "W": W}
