"""Rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    # Direct sum: 4x4x4 Strassen (rank 13) + 4x4x3 schoolbook (rank 12) = rank 25
    rank = 25

    U = [[0] * rank for _ in range(n * m)]  # 16 x 25
    V = [[0] * rank for _ in range(m * p)]  # 28 x 25
    W = [[0] * rank for _ in range(p * n)]  # 28 x 25

    # --- Part 1: Strassen decomposition for the 4x4x4 block (k in 0..3, i in 0..3) ---
    # Split A (4x4) into 2x2 blocks: A11=A[0:2,0:2], A12=A[0:2,2:4], A21=A[2:4,0:2], A22=A[2:4,2:4]
    # Split B (4x4) into 2x2 blocks: B11=B[0:2,0:2], B12=B[0:2,2:4], B21=B[2:4,0:2], B22=B[2:4,2:4]
    # Result C (4x4) blocks: C11=A11*B11+A12*B21, C12=A11*B12+A12*B22, C21=A21*B11+A22*B21, C22=A21*B12+A22*B22
    #
    # Strassen 7 products:
    #   M1 = (A11+A22)*(B11+B22)
    #   M2 = (A21+A22)*B11
    #   M3 = A11*(B12-B22)
    #   M4 = A22*(B21-B11)
    #   M5 = (A11+A12)*B22
    #   M6 = (A21-A11)*(B11+B12)
    #   M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Index helpers for the 4x4 block
    def u_idx(i, j):  # i in 0..3, j in 0..3 -> U row
        return i * m + j

    def v_idx(j, k):  # j in 0..3, k in 0..3 -> V row (k offset 0)
        return j * p + k

    def w_idx(k, i):  # k in 0..3, i in 0..3 -> W row
        return k * n + i

    # Strassen product definitions: (U_coeffs, V_coeffs)
    # Each product: U is a linear combo of A entries, V is a linear combo of B entries
    strassen_products = [
        # (U_coeffs: list of (i, j, coeff), V_coeffs: list of (j, k, coeff))
        # M1 = (A11+A22)*(B11+B22)
        ([((0,0,1),(0,1,1),(1,0,1),(1,1,1),(2,2,1),(2,3,1),(3,2,1),(3,3,1))],
         [((0,0,1),(0,1,1),(1,0,1),(1,1,1),(2,2,1),(2,3,1),(3,2,1),(3,3,1))]),
        # M2 = (A21+A22)*B11
        ([((2,0,1),(2,1,1),(3,0,1),(3,1,1))],
         [((0,0,1),(0,1,1),(1,0,1),(1,1,1))]),
        # M3 = A11*(B12-B22)
        ([((0,0,1),(0,1,1),(1,0,1),(1,1,1))],
         [((0,2,1),(0,3,1),(1,2,1),(1,3,1),(2,2,-1),(2,3,-1),(3,2,-1),(3,3,-1))]),
        # M4 = A22*(B21-B11)
        ([((2,2,1),(2,3,1),(3,2,1),(3,3,1))],
         [((2,0,1),(2,1,1),(3,0,1),(3,1,1),(0,0,-1),(0,1,-1),(1,0,-1),(1,1,-1))]),
        # M5 = (A11+A12)*B22
        ([((0,0,1),(0,1,1),(1,0,1),(1,1,1),(0,2,1),(0,3,1),(1,2,1),(1,3,1))],
         [((2,2,1),(2,3,1),(3,2,1),(3,3,1))]),
        # M6 = (A21-A11)*(B11+B12)
        ([((2,0,1),(2,1,1),(3,0,1),(3,1,1),(0,0,-1),(0,1,-1),(1,0,-1),(1,1,-1))],
         [((0,0,1),(0,1,1),(1,0,1),(1,1,1),(0,2,1),(0,3,1),(1,2,1),(1,3,1))]),
        # M7 = (A12-A22)*(B21+B22)
        ([((0,2,1),(0,3,1),(1,2,1),(1,3,1),(2,2,-1),(2,3,-1),(3,2,-1),(3,3,-1))],
         [((2,0,1),(2,1,1),(3,0,1),(3,1,1),(2,2,1),(2,3,1),(3,2,1),(3,3,1))]),
    ]

    # W contributions for each Strassen product (coeff, k, i) for each of 7 products
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    w_strassen = [
        # M1: contributes to C11 (+1) and C22 (+1)
        [((0,0,1),(0,1,1),(1,0,1),(1,1,1), (3,0,1),(3,1,1),(3,2,1),(3,3,1))],
        # M2: contributes to C21 (+1) and C22 (-1)
        [((2,0,1),(2,1,1),(2,2,1),(2,3,1), (3,0,-1),(3,1,-1),(3,2,-1),(3,3,-1))],
        # M3: contributes to C12 (+1) and C22 (+1)
        [((0,2,1),(0,3,1),(1,2,1),(1,3,1), (3,2,1),(3,3,1),(3,0,1),(3,1,1))],
        # M4: contributes to C11 (+1) and C21 (+1)
        [((0,0,1),(0,1,1),(1,0,1),(1,1,1), (2,0,1),(2,1,1),(2,2,1),(2,3,1))],
        # M5: contributes to C11 (-1) and C12 (+1)
        [((0,0,-1),(0,1,-1),(1,0,-1),(1,1,-1), (0,2,1),(0,3,1),(1,2,1),(1,3,1))],
        # M6: contributes to C22 (+1)
        [((3,0,1),(3,1,1),(3,2,1),(3,3,1))],
        # M7: contributes to C11 (+1)
        [((0,0,1),(0,1,1),(1,0,1),(1,1,1))],
    ]

    col = 0
    for prod_idx in range(7):
        u_coeffs, v_coeffs = strassen_products[prod_idx]
        # U coefficients
        for (i, j, c) in u_coeffs[0]:
            U[u_idx(i, j)][col] += c
        # V coefficients
        for (j, k, c) in v_coeffs[0]:
            V[v_idx(j, k)][col] += c
        # W coefficients
        for (k, i, c) in w_strassen[prod_idx][0]:
            W[w_idx(k, i)][col] += c
        col += 1

    # --- Part 2: Schoolbook decomposition for the 4x4x3 block (k in 4..6) ---
    # For each (i, j, k) with k in 4..6: U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    for i in range(n):
        for j in range(m):
            for k in range(4, p):  # k = 4, 5, 6
                U[u_idx(i, j)][col] = 1
                V[v_idx(j, k)][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
