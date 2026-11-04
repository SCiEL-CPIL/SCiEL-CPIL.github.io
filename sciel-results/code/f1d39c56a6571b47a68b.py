"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-42 decomposition of the 2x4x6 matrix multiplication tensor.

    Strategy: partition the 4x6 matrix B into three 4x2 blocks B_{1j}, B_{2j} for j=0,1,2.
    Each product C_{*j} = A * B_{*j} is a 2x4 * 4x2 multiplication.
    We further partition A into two 2x2 blocks A_{1*}, A_{2*}.
    Then C_{*j} = A_{11} * B_{1j} + A_{12} * B_{2j}.
    
    Each 2x2 * 2x2 product can be done with Strassen (rank 7).
    So each C_{*j} needs 14 terms (7 for A11*B1j + 7 for A12*B2j).
    Total rank = 3 * 14 = 42.

    Index conventions:
    A is n x m = 2 x 4, stored as U[i*m + j, c] where i in 0..1, j in 0..3.
    B is m x p = 4 x 6, stored as V[j*p + k, c] where j in 0..3, k in 0..5.
    C is p x n = 6 x 2, stored as W[k*n + i, c] where k in 0..5, i in 0..1.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42

    # Strassen's 7 terms for 2x2 matrices.
    # Each term: (a_indices, b_indices, c_indices) where each is a list of (row, col) for 2x2.
    # a_indices: positions in A_local (2x2) that are summed
    # b_indices: positions in B_local (2x2) that are summed  
    # c_indices: positions in C_local (2x2) where the product is placed (with sign)
    #
    # Strassen terms for C = A*B where A, B, C are 2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Each term is a list of (row, col, coeff) for the local 2x2 matrix
    strassen_terms = [
        # M1: (A11+A22)*(B11+B22) -> C11 +, C22 +
        ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),
        # M2: (A21+A22)*B11 -> C21 +, C22 -
        ([((1,0),1), ((1,1),1)], [((0,0),1)], [((1,0),1), ((1,1),-1)]),
        # M3: A11*(B12-B22) -> C12 +, C22 +
        ([((0,0),1)], [((0,1),1), ((1,1),-1)], [((0,1),1), ((1,1),1)]),
        # M4: A22*(B21-B11) -> C11 +, C21 +
        ([((1,1),1)], [((1,0),1), ((0,0),-1)], [((0,0),1), ((1,0),1)]),
        # M5: (A11+A12)*B22 -> C11 -, C12 +
        ([((0,0),1), ((0,1),1)], [((1,1),1)], [((0,0),-1), ((0,1),1)]),
        # M6: (A21-A11)*(B11+B12) -> C22 +
        ([((1,0),1), ((0,0),-1)], [((0,0),1), ((0,1),1)], [((1,1),1)]),
        # M7: (A12-A22)*(B21+B22) -> C11 +
        ([((0,1),1), ((1,1),-1)], [((1,0),1), ((1,1),1)], [((0,0),1)]),
    ]

    # Global index mapping:
    # A rows: i=0 (top), i=1 (bottom); cols: j=0,1 (block 1), j=2,3 (block 2)
    # A_local 2x2 for A_{11}: rows 0, cols 0,1 -> global A indices: i*m+j
    #   A_{11}[0,0] -> A[0,0] = U[0*4+0] = U[0]
    #   A_{11}[0,1] -> A[0,1] = U[1]
    #   A_{11}[1,0] -> A[1,0] = U[4]
    #   A_{11}[1,1] -> A[1,1] = U[5]
    # A_{12}: rows 0, cols 2,3
    #   A_{12}[0,0] -> A[0,2] = U[2]
    #   A_{12}[0,1] -> A[0,3] = U[3]
    #   A_{12}[1,0] -> A[1,2] = U[6]
    #   A_{12}[1,1] -> A[1,3] = U[7]

    # B columns: k=0,1 (block j=0), k=2,3 (block j=1), k=4,5 (block j=2)
    # B_{1j}: row 0 of B (j=0 in B's row index), cols 2j, 2j+1
    # B_{2j}: row 1 of B, cols 2j, 2j+1
    # B_local 2x2 for B_{1j}: 
    #   B_{1j}[0,0] -> B[0, 2j] = V[0*6+2j] = V[2j]
    #   B_{1j}[0,1] -> B[0, 2j+1] = V[2j+1]
    #   B_{1j}[1,0] -> B[1, 2j] = V[6+2j]
    #   B_{1j}[1,1] -> B[1, 2j+1] = V[6+2j+1]
    # B_{2j}:
    #   B_{2j}[0,0] -> B[2, 2j] = V[12+2j]
    #   B_{2j}[0,1] -> B[2, 2j+1] = V[12+2j+1]
    #   B_{2j}[1,0] -> B[3, 2j] = V[18+2j]
    #   B_{2j}[1,1] -> B[3, 2j+1] = V[18+2j+1]

    # C rows: k=0,1 (block j=0), k=2,3 (block j=1), k=4,5 (block j=2)
    # C_{*j} is a 2x2 where C_local[r,c] -> C[2j+r, c] where c is i (0 or 1)
    # W[k*n + i, col] where k = 2j + r, i = c
    #   C_local[0,0] -> C[2j, 0] = W[(2j)*2+0] = W[4j]
    #   C_local[0,1] -> C[2j, 1] = W[(2j)*2+1] = W[4j+1]
    #   C_local[1,0] -> C[2j+1, 0] = W[(2j+1)*2+0] = W[4j+2]
    #   C_local[1,1] -> C[2j+1, 1] = W[(2j+1)*2+1] = W[4j+3]

    def a11_global(local_r, local_c):
        return 0 * 4 + local_c  # i=0, j=local_c (0 or 1)

    def a12_global(local_r, local_c):
        return local_r * 4 + (2 + local_c)  # i=local_r, j=2+local_c

    def b1j_global(local_r, local_c, j):
        # B row 0+local_r, col 2j+local_c
        return (0 + local_r) * 6 + (2 * j + local_c)

    def b2j_global(local_r, local_c, j):
        # B row 2+local_r, col 2j+local_c
        return (2 + local_r) * 6 + (2 * j + local_c)

    def c_global(local_r, local_c, j):
        # C row 2j+local_r, col local_c
        return (2 * j + local_r) * 2 + local_c

    col = 0
    for j in range(3):  # three blocks of B
        for term_idx in range(7):
            a_indices, b_indices, c_indices = strassen_terms[term_idx]
            # First, A_{11} * B_{1j}
            for (ar, ac, a_coeff) in a_indices:
                U[a11_global(ar, ac)][col] = a_coeff
            for (br, bc, b_coeff) in b_indices:
                V[b1j_global(br, bc, j)][col] = b_coeff
            for (cr, cc, c_coeff) in c_indices:
                W[c_global(cr, cc, j)][col] = c_coeff
            col += 1

            # Second, A_{12} * B_{2j}
            for (ar, ac, a_coeff) in a_indices:
                U[a12_global(ar, ac)][col] = a_coeff
            for (br, bc, b_coeff) in b_indices:
                V[b2j_global(br, bc, j)][col] = b_coeff
            for (cr, cc, c_coeff) in c_indices:
                W[c_global(cr, cc, j)][col] = c_coeff
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
