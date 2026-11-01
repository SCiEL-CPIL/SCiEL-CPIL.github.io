"""Strassen rank-7 decomposition for 2x2 matrix multiplication."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """Return Strassen's rank-7 factor matrices for 2x2 matmul.

    Convention:
      U shape (n*m, rank) = (4, 7), indexed U[i*m + j, c]
      V shape (m*p, rank) = (4, 7), indexed V[j*p + k, c]
      W shape (p*n, rank) = (4, 7), indexed W[k*n + i, c]

    The matrix multiplication C = A @ B satisfies:
      C[k, i] = sum_j A[i, j] * B[j, k]

    In our tensor convention, the bilinear form for each rank-1 slice c is:
      U[:,c] outer V[:,c] contributes to W[:,c] in the sense that
      the product tensor T[a_idx, b_idx, c_idx] = sum_c U[a_idx,c]*V[b_idx,c]*W[c_idx,c]
      where a_idx = i*m+j, b_idx = j*p+k, c_idx = k*n+i.
    """
    n, m, p = DIMENSIONS
    rank = 7

    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # Results:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # U: row index = i*m + j  (i in 0..1, j in 0..1)
    #   row 0: i=0,j=0 -> A00
    #   row 1: i=0,j=1 -> A01
    #   row 2: i=1,j=0 -> A10
    #   row 3: i=1,j=1 -> A11
    U = [
        # A00: M1(+1), M2(0), M3(+1), M4(0), M5(+1), M6(-1), M7(0)
        [1, 0, 1, 0, 1, -1, 0],
        # A01: M1(0), M2(0), M3(0), M4(0), M5(+1), M6(0), M7(+1)
        [0, 0, 0, 0, 1, 0, 1],
        # A10: M1(0), M2(+1), M3(0), M4(0), M5(0), M6(+1), M7(0)
        [0, 1, 0, 0, 0, 1, 0],
        # A11: M1(+1), M2(+1), M3(0), M4(+1), M5(0), M6(0), M7(-1)
        [1, 1, 0, 1, 0, 0, -1],
    ]

    # V: row index = j*p + k  (j in 0..1, k in 0..1)
    #   row 0: j=0,k=0 -> B00
    #   row 1: j=0,k=1 -> B01
    #   row 2: j=1,k=0 -> B10
    #   row 3: j=1,k=1 -> B11
    V = [
        # B00: M1(+1), M2(+1), M3(0), M4(-1), M5(0), M6(+1), M7(0)
        [1, 1, 0, -1, 0, 1, 0],
        # B01: M1(0), M2(0), M3(+1), M4(0), M5(0), M6(+1), M7(0)
        [0, 0, 1, 0, 0, 1, 0],
        # B10: M1(0), M2(0), M3(0), M4(+1), M5(0), M6(0), M7(+1)
        [0, 0, 0, 1, 0, 0, 1],
        # B11: M1(+1), M2(0), M3(-1), M4(0), M5(+1), M6(0), M7(+1)
        [1, 0, -1, 0, 1, 0, 1],
    ]

    # W: row index = k*n + i  (k in 0..1, i in 0..1)
    #   row 0: k=0,i=0 -> C00
    #   row 1: k=0,i=1 -> C01
    #   row 2: k=1,i=0 -> C10
    #   row 3: k=1,i=1 -> C11
    W = [
        # C00 = M1 + M4 - M5 + M7
        [1, 0, 0, 1, -1, 0, 1],
        # C01 = M3 + M5
        [0, 0, 1, 0, 1, 0, 0],
        # C10 = M2 + M4
        [0, 1, 0, 1, 0, 0, 0],
        # C11 = M1 - M2 + M3 + M6
        [1, -1, 1, 0, 0, 1, 0],
    ]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
