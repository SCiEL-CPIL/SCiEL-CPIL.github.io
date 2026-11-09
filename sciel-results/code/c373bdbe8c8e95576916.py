def _make_2x2x2_strassen(nrows, ncolsm, ncolsk):
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1: (A00+A11)(B00+B11) -> C00
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1  # C00 only

    # M2: (A10+A11)B00 -> C11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[3, 1] = 1  # C11 only

    # M3: A00(B10-B11) -> C10, C11
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C10 (idx 1), C11 (idx 3)

    # M4: A11(B01-B00) -> C00, C11
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[3, 3] = 1  # C00 (idx 0), C11 (idx 3)

    # M5: (A00+A01)B11 -> C00, C01, C10
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1; W[1, 4] = 1  # C00 (idx 0), C01 (idx 2), C10 (idx 1)

    # M6: (A10-A00)(B00+B01) -> C01
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1  # C01 only (idx 2)

    # M7: (A01-A11)(B10+B11) -> C00
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C00 only (idx 0)

    return U, V, W
