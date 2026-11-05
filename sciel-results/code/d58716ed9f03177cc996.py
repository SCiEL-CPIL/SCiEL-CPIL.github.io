"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-17 exact decomposition of 2x3 * 3x3 matrix multiplication.
    
    Partition:
    - Strassen rank 7 for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    - Schoolbook rank 4 for A[0:2,2] * B[2,0:2] -> C[0:2,0:2]
    - Schoolbook rank 6 for A[0:2,0:3] * B[0:3,2] -> C[0:2,2]
    Total: 17
    
    Conventions:
    U[i*m+j, c]: i in [0,2), j in [0,3), row = i*3+j
    V[j*p+k, c]: j in [0,3), k in [0,3), row = j*3+k
    W[k*n+i, c]: k in [0,3), i in [0,2), row = k*2+i
    """
    n, m, p = 2, 3, 3
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms for 2x2x2 sub-block (i,j,k in {0,1})
    # Each term: list of (U_row, coeff), (V_row, coeff), (W_row, coeff)
    strassen = [
        # M1 = (A00+A11)(B00+B11)
        ([0, 4], [0, 4], {0: 1, 3: 1}),
        # M2 = (A10+A11)B00
        ([3, 4], [0], {1: 1, 3: -1}),
        # M3 = A00(B10-B11)
        ([0], [3, 4], {2: 1, 3: 1}),
        # M4 = A11(B01-B00)
        ([4], [1, 0], {0: 1, 1: 1}),
        # M5 = (A00+A01)B11
        ([0, 1], [4], {0: -1, 2: 1}),
        # M6 = (A10-A00)(B00+B10)
        ([3, 0], [0, 3], {3: 1}),
        # M7 = (A01-A11)(B01+B11)
        ([1, 4], [1, 4], {0: 1, 2: -1}),
    ]
    
    for u_rows, v_rows, w_coeffs in strassen:
        for ur in u_rows:
            U[ur][col] = 1
        for vr in v_rows:
            V[vr][col] = 1
        for wr, coeff in w_coeffs.items():
            W[wr][col] = coeff
        col += 1
    
    # Schoolbook: A[0:2,2] * B[2,0:2] -> C[0:2,0:2]
    # C[i,k] += A[i,2]*B[2,k] for i in {0,1}, k in {0,1}
    for i in range(2):
        for k in range(2):
            U[i * 3 + 2][col] = 1
            V[2 * 3 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    
    # Schoolbook: A[0:2,0:3] * B[0:3,2] -> C[0:2,2]
    # C[i,2] += A[i,j]*B[j,2] for i in {0,1}, j in {0,1,2}
    for i in range(2):
        for j in range(3):
            U[i * 3 + j][col] = 1
            V[j * 3 + 2][col] = 1
            W[2 * 2 + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

