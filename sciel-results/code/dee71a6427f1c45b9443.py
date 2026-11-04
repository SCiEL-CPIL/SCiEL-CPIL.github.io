"""Rank-24 decomposition for 3x4x4 matrix multiplication.

Splits the 3x4x4 product by rows:
  - rows 0,1 form a 2x4x4 block handled by Strassen (rank 16)
  - row 2 forms a 1x4x4 block handled by schoolbook (rank 8)
Total rank = 24.
"""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 4
    # U: (n*m, r) indexed [i*m + j]
    # V: (m*p, r) indexed [j*p + k]
    # W: (p*n, r) indexed [k*n + i]
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # ---- Strassen block for rows i=0,1 ----
    # Split columns into A = [A11 A12] (j=0,1 | j=2,3)
    # Split rows of B into B = [B11; B12] (k=0,1 | k=2,3)
    # C11 = A11*B11, C12 = A11*B12, C21 = A12*B11, C22 = A12*B12
    # Strassen: M1=(A11+A12)(B11+B12), M2=A12(B11+B12), M3=A11(B12-B11),
    #           M4=A12(B12-B11), M5=(A11+A12)B12, M6=(A12-A11)B11
    # C11=M3+M4+M5+M6, C12=M1-M5, C21=M2-M4, C22=M1-M2+M3
    strassen = [
        # (u_coef, v_coef, [(i,k,w_coef), ...])
        ((1, 1), (1, 1), [(-1, -1, -1), (-1, 0, 1), (1, -1, -1), (1, 0, 1)]),  # M1
        ((1, 0), (1, 1), [(-1, -1, -1), (0, -1, 1), (1, -1, 1), (1, 0, -1)]),   # M2
        ((1, 0), (0, -1), [(-1, -1, 1), (-1, 0, 1), (1, -1, -1), (1, 0, 1)]),   # M3
        ((1, 0), (0, -1), [(-1, -1, -1), (0, -1, 1), (1, -1, -1), (1, 0, 1)]),  # M4
        ((1, 1), (0, 1), [(-1, -1, 1), (1, -1, 1)]),                            # M5
        ((1, -1), (0, 1), [(-1, -1, 1), (1, -1, -1)]),                          # M6
    ]
    col = 0
    for (u11, u12), (v11, v12), out in strassen:
        # U entries: row i*m+j
        if u11:
            U[0 * m + 0][col] += u11  # i=0,j=0
            U[0 * m + 1][col] += u11  # i=0,j=1
        if u12:
            U[0 * m + 2][col] += u12  # i=0,j=2
            U[0 * m + 3][col] += u12  # i=0,j=3
            U[1 * m + 2][col] += u12  # i=1,j=2
            U[1 * m + 3][col] += u12  # i=1,j=3
        # V entries: row j*p+k
        if v11:
            V[0 * p + 0][col] += v11  # j=0,k=0
            V[1 * p + 0][col] += v11  # j=1,k=0
            V[0 * p + 1][col] += v11  # j=0,k=1
            V[1 * p + 1][col] += v11  # j=1,k=1
        if v12:
            V[0 * p + 2][col] += v12  # j=0,k=2
            V[1 * p + 2][col] += v12  # j=1,k=2
            V[0 * p + 3][col] += v12  # j=0,k=3
            V[1 * p + 3][col] += v12  # j=1,k=3
        # W entries: row k*n+i
        for (i, k, wcoef) in out:
            W[k * n + i][col] += wcoef
        col += 1

    # ---- Schoolbook for row i=2 ----
    # C[i,k] = sum_j A[2,j]*B[j,k], j=0..3, k=0..3
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
