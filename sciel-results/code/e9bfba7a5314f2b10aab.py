"""Rewritten program for matrix multiplication 2x3x5 tensor decomposition.

Strategy: Use Strassen's 7-multiplication algorithm for the 2x2 sub-block
(j in {1,2}, k in {1,2}), and schoolbook for the remaining terms.
This gives rank 7 + 12 + 10 = 29, saving 1 rank over pure schoolbook (30).
"""

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2 rows of A, m=3 cols of A / rows of B, p=5 cols of B
    # A is n x m = 2x3, B is m x p = 3x5, C is n x p = 2x5
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (15, rank)
    # W shape: (p*n, rank) = (10, rank)  [W indexed as W[k*n + i, r]]

    U_entries = []  # list of (row, val) pairs per rank term
    V_entries = []
    W_entries = []

    # Strassen for the 2x2 sub-block: A_sub is rows i=0,1 and cols j=0,1
    # B_sub is rows j=0,1 and cols k=0,1
    # C_sub is rows i=0,1 and cols k=0,1
    
    # Strassen's 7 products for 2x2x2:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    strassen_products = [
        # (u_coefficients, v_coefficients, w_coefficients)
        # Each coefficient is a dict mapping (i,j) or (j,k) or (k,i) to value
        # M1: (a00+a11)*(b00+b11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2: (a10+a11)*b00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 1, (1,0): 1}),
        # M3: a00*(b01-b11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,0): 1}),
        # M4: a11*(b10-b00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): -1, (1,0): 1}),
        # M5: (a00+a01)*b11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6: (a10-a00)*(b00+b01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7: (a01-a11)*(b10+b11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    for u_dict, v_dict, w_dict in strassen_products:
        U_entries.append([(i, j), v] for (i, j), v in u_dict.items())
        V_entries.append([(j, k), v] for (j, k), v in v_dict.items())
        # W is indexed as k*n + i
        W_entries.append([(k, i), v] for (k, i), v in w_dict.items())

    # Schoolbook for j=0,1 and k=2,3,4 (the remaining k columns for the first two rows)
    # 2 * 3 = 6 terms
    for i in range(2):
        for j in range(2):
            for k in range(2, p):
                U_entries.append([(i, j), 1])
                V_entries.append([(j, k), 1])
                W_entries.append([(k, i), 1])

    # Schoolbook for j=2 and k=0,1,2,3,4 (the third column of A with all columns of B)
    # 2 * 5 = 10 terms
    for i in range(n):
        for k in range(p):
            U_entries.append([(i, 2), 1])
            V_entries.append([(2, k), 1])
            W_entries.append([(k, i), 1])

    rank = len(U_entries)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for (i, j), val in U_entries[r]:
            U[i * m + j][r] = val
        for (j, k), val in V_entries[r]:
            V[j * p + k][r] = val
        for (k, i), val in W_entries[r]:
            W[k * n + i][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
