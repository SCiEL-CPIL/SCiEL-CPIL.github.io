"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # s=0: M1 = (A00+A11)(B00+B11)
    # s=1: M2 = (A10+A11)B00
    # s=2: M3 = A00(B01-B11)
    # s=3: M4 = A11(B10-B00)
    # s=4: M5 = (A00+A01)B11
    # s=5: M6 = (A10-A00)(B00+B01)
    # s=6: M7 = (A01-A11)(B10+B11)
    
    # alpha[s][I][J] is coeff of A[I][J] in factor 1 of M_s
    # beta[s][J][K] is coeff of B[J][K] in factor 2 of M_s
    # gamma[s][I][K] is coeff of M_s in C[I][K]
    
    # Using lists of lists for 2x2 matrices
    # Indexing [row][col]
    
    alpha = [
        [[1, 0], [0, 1]],  # s=0
        [[0, 0], [1, 1]],  # s=1
        [[1, 0], [0, 0]],  # s=2
        [[0, 0], [0, 1]],  # s=3
        [[1, 1], [0, 0]],  # s=4
        [[-1, 0], [1, 0]], # s=5
        [[0, 1], [0, -1]]  # s=6
    ]
    
    beta = [
        [[1, 0], [0, 1]],  # s=0
        [[1, 0], [0, 0]],  # s=1
        [[0, 1], [0, -1]], # s=2
        [[-1, 0], [0, 1]], # s=3
        [[0, 0], [0, 1]],  # s=4
        [[1, 1], [0, 0]],  # s=5
        [[0, 0], [1, 1]]   # s=6
    ]
    
    gamma = [
        [[1, 0], [0, 1]],  # s=0: C00 += M1, C11 += M1
        [[0, 0], [1, 0]],  # s=1: C10 += M2
        [[0, 1], [0, 0]],  # s=2: C01 += M3
        [[1, 0], [1, 0]],  # s=3: C00 += M4, C10 += M4
        [[-1, 0], [0, 0]], # s=4: C00 -= M5
        [[0, 0], [0, 1]],  # s=5: C11 += M6
        [[1, 0], [0, 0]]   # s=6: C00 += M7
    ]
    
    # Construct U, V, W
    # r = 7*s + t
    for s in range(7):
        for t in range(7):
            r = 7 * s + t
            # Iterate over block indices I, J, K (0 or 1)
            # and local indices a, b, c (0 or 1)
            for I in range(2):
                for J in range(2):
                    for K in range(2):
                        for a in range(2):
                            for b in range(2):
                                for c in range(2):
                                    i = 2 * I + a
                                    j = 2 * J + b
                                    k = 2 * K + c
                                    
                                    u_val = alpha[s][I][J] * alpha[t][a][b]
                                    v_val = beta[s][J][K] * beta[t][b][c]
                                    w_val = gamma[s][I][K] * gamma[t][a][c]
                                    
                                    if u_val != 0:
                                        U[i * m + j][r] += u_val
                                    if v_val != 0:
                                        V[j * p + k][r] += v_val
                                    if w_val != 0:
                                        W[k * n + i][r] += w_val
                                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
